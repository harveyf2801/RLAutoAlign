from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.logger import TensorBoardOutputFormat

import os
import numpy as np
import time


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