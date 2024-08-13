<!--
 * @Author: LOTEAT
 * @Date: 2024-08-12 21:44:55
-->
## Scheduler 
- 前置知识：PyTorch
- [代码链接](https://github.com/open-mmlab/mmengine)


### 1. Code
本文主要介绍`mmengine`中的`_ParamScheduler`类。在`_ParamScheduler`中，核心函数就是`step`函数，讲解在代码注释中。
```python
    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after '
                    'parameter value scheduler initialization. Please, make '
                    'sure to call `optimizer.step()` before '
                    '`scheduler.step()`. See more details at '
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before '
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you '
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        # 全局步长计算，每一个iter/epoch都会加1
        self._global_step += 1

        # 如果self._global_step在self.begin和self.end之中
        # 调整是从[self.begin, self.end) iter之间调整
        if self.begin <= self._global_step < self.end:
            # self.last_step用于记录最新的step
            # self.last_step在resuming模型的时候还会被用到
            self.last_step += 1
            # 通过scheduler的策略对optimizer的params_groups更新
            values = self._get_value()
            # 更新
            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)
        # self._last_value进行记录
        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]
```