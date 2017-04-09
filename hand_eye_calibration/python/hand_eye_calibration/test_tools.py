import numpy as np
from random import randint
import logging


class DataDropConfig:
  overall_drop_percentage = 20.0
  max_percentage_for_single_drop = 20.0

  def __init__(self):
    pass


def introduce_data_drops(data, drop_config, set_to_none=False):
  assert isinstance(drop_config, DataDropConfig)
  assert type(data) is list
  assert drop_config.max_percentage_for_single_drop < drop_config.overall_drop_percentage

  num_datapoints = len(data)

  max_datapoints_per_drop = (drop_config.max_percentage_for_single_drop /
                             100.0) * float(num_datapoints)
  overall_num_datapoints_to_drop = (
      drop_config.overall_drop_percentage / 100.0) * float(num_datapoints)

  print("Indroducing data drops for a dataset of length {}, "
        "overall number of datapoints to drop: {} max number "
        "of datapoints per drop: {}.".format(num_datapoints,
                                             overall_num_datapoints_to_drop,
                                             max_datapoints_per_drop))

  current_num_datapoints_dropped = 0
  dropped_datapoints_set = set()
  while (current_num_datapoints_dropped < overall_num_datapoints_to_drop):
    remaining_datapoints_to_drop = overall_num_datapoints_to_drop - current_num_datapoints_dropped

    drop_start = randint(0, num_datapoints - 1)
    drop_size = randint(0, min(remaining_datapoints_to_drop, max_datapoints_per_drop))
    drop_end = min(drop_start + drop_size, num_datapoints - 1)

    dropped_datapoints_set.update(range(drop_start, drop_end))

    current_num_datapoints_dropped = len(dropped_datapoints_set)

  assert len(dropped_datapoints_set) == overall_num_datapoints_to_drop

  for i in sorted(dropped_datapoints_set, reverse=True):
    if set_to_none:
      data[i] = None
    else:
      del data[i]
  print("Dropped {} datapoints.".format(len(dropped_datapoints_set)))
