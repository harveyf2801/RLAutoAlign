from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.logger import TensorBoardOutputFormat

import os
import numpy as np
import time

class SaveOnBestTrainingRewardCallback(BaseCallback):
  """
  Callback for saving a model (the check is done every ``check_freq`` steps)
  based on the training reward (in practice, we recommend using ``EvalCallback``).

  :param check_freq:
  :param log_dir: Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
  :param verbose: Verbosity level.
  """
  def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
    super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.log_dir = log_dir
    self.save_path = os.path.join(log_dir, 'best_model')
    self.best_mean_reward = -np.inf

  def _init_callback(self) -> None:
      # Create folder if needed
      if self.save_path is not None:
          os.makedirs(self.save_path, exist_ok=True)

  def _on_step(self) -> bool:
      if self.n_calls % self.check_freq == 0:

        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
              print(f"Num timesteps: {self.num_timesteps}")
              print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                  print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

      return True

class SaveEveryHourCallback(BaseCallback):
  """
  Callback for saving a model every hour.

  :param log_dir: Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
  :param verbose: Verbosity level.
  """
  def __init__(self, log_dir: str, verbose: int = 1):
    super(SaveEveryHourCallback, self).__init__(verbose)
    self.log_dir = log_dir
    self.save_path = os.path.join(log_dir, 'auto_save_model')
    self.last_save_time = time.time()

  def _on_step(self) -> bool:
    current_time = time.time()
    elapsed_time = current_time - self.last_save_time

    if elapsed_time >= 3600:  # 3600 seconds = 1 hour
      self.last_save_time = current_time

      if self.verbose > 0:
        print(f"Saving model at {self.save_path}")

      self.model.save(self.save_path)

    return True

class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 100  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals['my_custom_info_dict']['my_custom_reward']
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("rewards/env #{}".format(i+1),
                                                     rewards[i],
                                                     self.n_calls)