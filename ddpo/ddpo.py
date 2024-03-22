import math
import torch

from fastprogress import progress_bar
from ddpo.utils import flush


EPS = 1e-6

standardize = lambda x: (x - x.mean()) / (x.std() + EPS)

def calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t, eps=EPS):
  """Compute logs probs for prev_sample from a normal distribution with mean
  prev_sample_meand and std std_dev_t"""
  std_dev_t = torch.clip(std_dev_t, eps)
  log_probs = (
      -(prev_sample.detach() - prev_sample_mean)**2 / (2 * std_dev_t**2)
      - torch.log(std_dev_t)
      - math.log(math.sqrt(2 * math.pi))
  )
  return log_probs


@torch.no_grad()
def sample_from_ddpm_celebahq(num_samples,
                                    scheduler,
                                    image_pipe,
                                    reward_model,
                                    device,
                                    eta=1,
                                    random_seed=None):
  """
  This function samples a batch of images from the google/ddpm-celebahq-256 model
  using a specified scheduler, image pipeline, reward model, and device. It generates
  random noise as an initial input and iteratively updates the noise based on the
  scheduler's output. The function computes rewards for each updated noise sample
  using the provided reward model. The trajectory of noise samples and their
  corresponding rewards are saved in the output dictionary.

  Reference of diffuser sample loop: https://huggingface.co/blog/stable_diffusion

  Args:
    num_samples (int): The number of samples to generate.
    scheduler (DDIMScheduler): The scheduler object that controls the sampling process.
    image_pipe (ImagePipeline): The image pipeline object used for processing images.
    reward_model (AestheticRewardModel): The reward model used to compute the aesthetic score of the samples.
    device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform computations.
    random_seed (int, optional): The random seed for reproducibility. Defaults to None.

  Returns:
    tensor: A tensor containing the trajectories of the entire batach (T, B, C, H, w).
    tensor: A tensor containing the log probabilities of the trajectories (T, B).
    tensor: A tensor containing the final reward (from image generated) computed
     using the reward model (B, 1)
  """
  if random_seed:
    torch.manual_seed(random_seed)

  num_inference_steps = scheduler.num_inference_steps

  # initialize a batch of random noise
  xt = torch.randn(num_samples, 3, 256, 256).to(device)   # (B, C, H, W)

  # save initial state x_T and intermediate steps, saave log_probs for the trajectory
  trajectory, log_probs = [xt], []

  for i, t in (enumerate(progress_bar(scheduler.timesteps))):
      # [S] scale input based on the timestep
      model_input = scheduler.scale_model_input(xt, timestep=t)

      # [S] get the noise prediction (unet predicts noise residual)
      noise_pred = image_pipe.unet(model_input, t).sample

      # [S] using the prediction noise we can predict the denoised image representation
      # compute the "previous" noisy sample mean 
      scheduler_output = scheduler.step(noise_pred, t, xt, eta, variance_noise=0)
      prev_sample_mean = scheduler_output.prev_sample # this is the mean and not full sample since variance is 0

      # [S] Computa la varianza entre los dos timesteps actual y anterior,
      # se debe considerar los saltos entre timesteps de entrenamiento e inferencia.
      t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
      variance = scheduler._get_variance(t, t_1)
      std_dev_t = eta * variance ** (0.5)

      # [S] generamos nuevas muestras (re-parametrization trick)
      prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t # get full sample by adding noise

      # [S] compute the log probs of the new sample
      log_probs.append(calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim))))

      trajectory.append(prev_sample)
      xt = prev_sample

  # compute final reward and save it; save trajectory
  final_reward = reward_model(xt)

  # now we will release the VRAM memory deleting the variable bounded to the VRAM
  # and use flush()
  del xt
  del model_input
  del noise_pred
  del scheduler_output
  del prev_sample_mean
  del prev_sample
  del variance
  del std_dev_t
  del num_inference_steps
  flush()

  # (T+1, B, C, H, W), (T, B), (B, 1)
  return torch.stack(trajectory), torch.stack(log_probs), final_reward



def compute_loss(x_t,
                 original_log_probs,
                 advantages,
                 clip_advantages,
                 clip_ratio,
                 image_pipe,
                 scheduler,
                 device,
                 eta=1):
  """Compute DDPO_is loss for a batch of samples"""
  unet = image_pipe.unet.to(device)
  num_inference_steps = scheduler.num_inference_steps
  loss_value = 0.
  for i, t in enumerate(scheduler.timesteps):
    clipped_advantages = torch.clip(advantages, -clip_advantages, clip_advantages).detach()

    # scale the input by the current timestep t and predict the noise residual
    input = scheduler.scale_model_input(x_t[i].detach(), t)
    pred = unet(input, t).sample

    # compute the "previous" noisy sample mean and variance, and get log probs
    scheduler_output = scheduler.step(pred, t, x_t[i].detach(), eta, variance_noise=0)
    t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
    variance = scheduler._get_variance(t, t_1)
    std_dev_t = eta * variance ** (0.5)
    prev_sample_mean = scheduler_output.prev_sample
    current_log_probs = calculate_log_probs(x_t[i+1].detach(), prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

    # calculate loss
    ratio = torch.exp(current_log_probs - original_log_probs[i].detach()) # this is the importance ratio of the new policy to the old policy
    unclipped_loss = -clipped_advantages * ratio # this is the surrogate loss
    clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - clip_ratio, 1. + clip_ratio) # this is the surrogate loss, but with artificially clipped ratios
    loss = torch.max(unclipped_loss, clipped_loss).mean() # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
    loss.backward() # perform backward here, gets accumulated for all the timesteps

    loss_value += loss.item()
  return loss_value