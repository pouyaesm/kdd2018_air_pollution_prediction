import numpy as np
import pandas as pd
from src import util
from datetime import datetime, timedelta
from collections import deque


def group(ts: pd.DataFrame, mode, value=None, time_key='time', agg_op=None):
    """
        Groups data into specific time intervals, then returns aggregated values of
        those with f(time) = value; both intervals and matching rows depend on the given mode
        modes:
            h: hour of day [0, 23]
            dw: day of week [0, 6], where 0: sunday, 6: saturday
            w: week of month [0, 3]
            m: month of year [1, 12]
            s: season of year [1, 3]
        dt.<fields>: https://pandas.pydata.org/pandas-docs/stable/api.html#datetimelike-properties
        Prepare time series data for learning
    """
    agg_op = {'value': 'mean'} if agg_op is None else agg_op
    sample = ts.copy()
    time = pd.to_datetime(ts[time_key])
    if mode == 'h':
        sample[time_key] = time.dt.strftime("%Y-%m-%d %H:00:00")  # coarse-grain the time to hour of day
        matches_value = time.dt.hour == value
    elif mode == '3h':
        time = time.apply(lambda dt: round_hour(dt, 3))
        sample[time_key] = time.dt.strftime("%Y-%m-%d %H:00:00")
        matches_value = time.dt.hour == value
    elif mode == '6h':
        time = time.apply(lambda dt: round_hour(dt, 6))
        sample[time_key] = time.dt.strftime("%Y-%m-%d %H:00:00")
        matches_value = time.dt.hour == value
    elif mode == 'dw':
        sample[time_key] = time.dt.strftime("%Y-%W-%w")  # coarse-grain the time to day of week
        # in pandas dayofweek is 0: monday, 6: sunday,
        # unlike strftime where 0: sunday, 6:saturday
        matches_value = time.dt.dayofweek == ((value + 6) % 7)
    elif mode == 'm':
        sample[time_key] = time.dt.strftime("%Y-%m")  # coarse-grain the time to month
        matches_value = time.dt.month == value
    elif mode == 's':
        sample[time_key] = np.mod(time.dt.strftime('%Y-%m').astype(int), 4)

    # sample those with matching time values
    sample = sample.loc[matches_value, :] if value is not None else sample
    # aggregate values based on mode
    sample = sample.groupby([time_key], as_index=False).agg(agg_op)
    sample.reset_index(drop=True, inplace=True)
    return sample


def group_at(time: list, value: list, index, direction, group_hours):
    aggregate = 0
    count = 0
    max_index = len(time) - 1  # max allowed index
    cur_t = time[index]
    round_t = group_t = round_hour(cur_t, group_hours)
    while group_t == round_t:
        # t: rounded time to aggregate values using a dictionary
        aggregate += value[index]
        count += 1
        index = index + direction
        if index < 0 or index > max_index:
            break
        cur_t = time[index]
        round_t = round_hour(cur_t, group_hours)

    return aggregate / count if count > 0 else 0


def group_from(time: list, value: list, index, step, group_hours, whole_group=False):
    """
        Group values according to unit, that are mostly hours * step behind the time(index)
    :param time: time series of data-time objects
    :param value:   time series of values to be grouped
    :param index: index of origin value to start the grouping (backward or forward)
    :param step: negative index for backward, positive for forward (0 for current group)
    :param group_hours: number of hours as a unit to group
    :param whole_group: if True the grouping exceeds the index after (when step < 0)
        or before (when step > 0) to calculate the total group average
    :return:
    """
    aggregate = dict()  # sum of values for a time group
    count = dict()  # number of values for a time group
    direction = np.sign(step)  # positive is forward
    max_index = len(time) - 1  # max allowed index
    cur_t = time[index]
    group_t = round_hour(cur_t, group_hours)
    round_t = None
    if direction > 0:
        # for groups [3, 6)[6, 9) and group('5') = '3' and step 1, go from '5' to '8'
        limit_t = group_t + timedelta(hours=(abs(step)) * group_hours)
    else:
        # for groups [3, 6)[6, 9) and group('7') = '6' and step 1, go from '3' to '7'
        limit_t = group_t - timedelta(hours=abs(step - 1) * group_hours)
    # Start from the start of group  that includes 'index' (when step > 0)
    # Start from the end of group that includes 'index' (when step < 0)
    if whole_group:
        round_t = round_hour(cur_t, group_hours)
        # keep moving the index until passing the group boundary
        while round_t == group_t:
            index = index - direction
            if index < 0 or index > max_index:
                break
            cur_t = time[index]
            round_t = round_hour(cur_t, group_hours)
        # undo the last change in index
        index = index + direction
        cur_t = time[index]

    while (direction > 0 and cur_t < limit_t) or (direction < 0 and cur_t >= limit_t):
        # t: rounded time to aggregate values using a dictionary
        round_t = round_hour(cur_t, group_hours)
        aggregate[round_t] = aggregate[round_t] + value[index] \
            if round_t in aggregate else value[index]
        count[round_t] = count[round_t] + 1 if round_t in count else 1
        index = index + direction
        if index < 0 or index > max_index:
            break
        cur_t = time[index]

    # remained_steps > 0 when index hits array boundary before "step" entries are filled
    remained_steps = abs(step) - len(aggregate)
    if remained_steps > 0:
        for i in range(0, remained_steps):
            cur_t = cur_t + direction * timedelta(hours=group_hours)
            t = round_hour(cur_t, group_hours)
            aggregate[t] = aggregate[round_t]
            count[t] = count[round_t]

    for t, v in iter(aggregate.items()):
        aggregate[t] /= count[t]

    return [value for (key, value) in sorted(aggregate.items())]


def running_average(time: list, value: list, group_hours: int, direction = 1, whole_group=False):
    """
        Put average values at each index by aggregating values
        from current time group of duration 'hours' until that value
    :param time: time series of data-time objects
    :param value:  time series of values to be grouped
    :param group_hours: number of hours of each time group
    :param direction: if positive start from 0, otherwise start from last element
    :param whole_group: assign average of whole time group for its members
        not just the running average until an index
    :return:
    """
    size = len(time)
    if size != len(value):  # each value must have a corresponding time
        return -1

    run_average = [0] * size  # average of values at index i corresponding to current time group
    aggregate = dict()  # sum of values for a time group
    count = dict()  # number of values for a time group

    t_group_pre = round_hour(time[0], group_hours)
    start_index = 0
    # index goes one extra step to account for the last group when whole_group = True
    for index in range(0, size + 1):
        if direction > 0:
            i = index
        else:  # compute the running average from last to first element
            i = size - index - 1 if index < size else 0
        if index < size:
            t_group = round_hour(time[i], group_hours)  # round time to 'hours' unit

            aggregate[t_group] = aggregate[t_group] + value[i] \
                if t_group in aggregate else value[i]
            count[t_group] = count[t_group] + 1 if t_group in count else 1
            if not whole_group:
                # put average of group util index for element in this position
                run_average[i] = aggregate[t_group] / count[t_group]

        if whole_group and (t_group != t_group_pre or index == size):
            # put average of whole group for all elements
            average = aggregate[t_group_pre] / count[t_group_pre]
            if direction > 0:
                run_average[start_index:i] = [average] * (i - start_index)
            else:
                run_average[i:start_index] = [average] * (start_index - i)
            # collect average of next (coming) group
            t_group_pre = t_group
            start_index = i

    return run_average


def group_average(time: list, value: list, group_hours: int):
    """
        Put average values at each index by aggregating values
        from current time group of duration 'hours' until that value
    :param time: time series of data-time objects
    :param value:  time series of values to be grouped
    :param group_hours: number of hours of each time group
    :return:
    """
    size = len(time)
    if size != len(value):  # each value must have a corresponding time
        return -1

    aggregate = list()  # sum of values for a time group
    count = list()  # number of values for a time group
    group_time = list()  # time corresponding to each group average
    group_lookup = dict()  # for accessing the group average from group time

    t_group_pre = -1
    group_index = -1
    for i in range(0, size):
        t_group = round_hour(time[i], group_hours)  # round time to 'hours' unit
        if t_group != t_group_pre:
            group_index += 1
            aggregate.append(0)
            count.append(0)
            group_time.append(t_group)
            group_lookup[t_group] = group_index
            t_group_pre = t_group
        aggregate[group_index] = aggregate[group_index] + value[i]
        count[group_index] += 1

    for i in range(0, len(aggregate)):
        aggregate[i] = aggregate[i] / count[i] if count[i] > 0 else 0

    return group_time, aggregate, group_lookup, count

# def split(time: list, value: list, hours, step, skip=0, whole_group=False):
#     """
#         Split and group 'step' number of averaged values 'hours' apart
#     :param time: time per value (hour apart)
#     :param value: assumed to have value[step * t - 1] = average[value[step * (t - 1):step * t]
#     :param hours: group times into 'hours' hours
#     :param step: number of group times set for each index
#     :param skip: ignore offset number of first values
#     :param whole_group: include the aggregated value of
#     whole time group for each of its members not just until that member
#     :return:
#     """
#     splits = list()  # step group times per index
#     size = len(time)
#     if size != len(value):
#         return -1
#     # Calculate running average of values
#     # that resets by entering a new time group of duration 'hours'
#     run_average = running_average(time=time, value=value,
#                                   hours=hours, whole_group=whole_group)
#     # array of last 'step' averages to be set for on-going index
#     cur_step = 0
#     step_values = [0] * step
#     t_group_pre = round_hour(time[0], hours)  # first time group to begin
#     for index, value in enumerate(run_average):
#         t_group = round_hour(time[index], hours)
#         if t_group != t_group_pre:  # entering new time group
#             t_group_pre = t_group
#             if cur_step < step - 1:
#                 cur_step = min(cur_step + 1, step - 1)
#             else:
#                 util.shift(step_values)  # shift array toward 0
#
#         # Set running average of current time group on ward
#         if cur_step < step - 1:
#             step_values[cur_step:] = [value] * (step - cur_step)
#         else:
#             step_values[cur_step] = value
#
#         if index < skip:
#             continue  # skip first 'offset' indices
#         # Set 'step' count averaged values for current index
#         splits.append(step_values.copy())
#     return splits


# def split(time: list, value: list, step, group_hours, region=None, whole_group=False):
#     """
#         Split and group 'step' number of averaged values 'hours' apart
#     :param time: time per value (hour apart)
#     :param value: values corresponding to time
#     :param step: number of group times set for each index
#     :param group_hours: group times into 'hours' hours
#     :param region: region of indices to be considered
#     :param whole_group: include the aggregated value of
#     whole time group for each of its members not just until that member
#     :return:
#     """
#     splits = list()  # step group times per index
#     size = len(time)
#     if size != len(value):
#         return -1
#     # direction is the sign of step
#     direction = np.sign(step)
#     # indices to be considered
#     region = (0, size) if region is None else region
#     i_range = range(max(region[0], 0), size if region[1] < 0 else region[1])
#     # Running group average of each index either forward (when step < 0)
#     # or backward (when step > 0), when whole_group = False
#     if not whole_group:
#         run_average = running_average(time, value, group_hours=group_hours,
#                                       direction=-np.sign(step), whole_group=False)
#     else:
#         run_average = []
#     group_time, average, group_lookup, _ = group_average(time, value, group_hours=group_hours)
#     group_size = len(group_time)
#
#     # duplicated first (for forward) or last (for backward) group average as array of step values
#     step_values = [average[0 if direction > 0 else -1]] * abs(step)
#
#     pre_group_index = group_lookup[round_hour(time[i_range[0]], group_hours)]
#     for i in i_range:
#         group_index = group_lookup[round_hour(time[i], group_hours)]
#         last_index = group_index + step - direction
#         if step > 0:  # forward grouping
#             # repeat the last group average if step goes outside the group array
#             if last_index < group_size:
#                 step_values
#             step_values = average[group_index:last_index + 1] \
#                 if last_index < group_size \
#                 else average[group_index:] + ([average[-1]] * (last_index - group_size + 1))
#         else:
#             # repeat the first group average if step goes outside the group array
#             step_values = average[last_index:group_index + 1] \
#                 if last_index > 0 else ([average[0]] * -last_index) + average[0:group_index + 1]
#         # replace the group average with partial average if the whole group is not required
#         if not whole_group:
#             step_values[0 if step > 0 else -1] = run_average[i]
#
#         splits.append(step_values)
#     return splits


def split(time: list, value: list, step, group_hours, region=None, whole_group=False):
    """
        Split and group 'step' number of averaged values 'hours' apart
    :param time: time per value (hour apart)
    :param value: values corresponding to time
    :param step: number of group times set for each index
    :param group_hours: group times into 'hours' hours
    :param region: region of indices to be considered
    :param whole_group: include the aggregated value of
    whole time group for each of its members not just until that member
    :return:
    """
    splits = list()  # step group times per index
    size = len(time)
    if size != len(value):
        return -1
    # direction is the sign of step
    direction = np.sign(step)
    # indices to be considered
    region = (0, size - 1) if region is None else region
    region = (max(region[0], 0), size - 1 if region[1] < 0 else region[1])
    # Running group average of each index either forward (when step < 0)
    # or backward (when step > 0), when whole_group = False
    if not whole_group:
        run_average = running_average(time, value, group_hours=group_hours,
                                      direction=-np.sign(step), whole_group=False)
    else:
        run_average = []
    group_time, average, group_lookup, _ = group_average(time, value, group_hours=group_hours)
    group_size = len(group_time)

    # init first 'steps' (for forward)
    # or duplication o first (for backward) [whole/partial] group average as array of step values
    group_time = pre_group_time = round_hour(time[region[0]], group_hours)
    group_index = group_lookup[group_time]
    last_index = group_index + step - direction
    if step > 0:
        initial_values = average[group_index:min(last_index + 1, group_size)]
        if len(initial_values) != abs(step):  # duplicate the last group average to reach 'step' values
            initial_values += [[average[-1] * (group_size - last_index)]]
    else:
        initial_values = average[max(last_index, 0):group_index + 1]
        if len(initial_values) != abs(step):  # duplicate the first group average to reach 'step' values
            initial_values = ([average[0]] * (-last_index)) + initial_values

    step_values = deque(initial_values)

    cur_step = 0
    for i in range(region[0], region[1] + 1):
        group_time = round_hour(time[i], group_hours)
        if group_time != pre_group_time:
            group_index = group_lookup[group_time]
            last_index = group_index + step - direction
            cur_step = min(step, cur_step + 1)
            step_values.rotate(-1)  # shift right to go toward end of groups
            # duplicate the second to last value if group size is passed
            # otherwise set the last value from group averages
            if step > 0:
                step_values[-1] = average[last_index] if last_index < group_size else step_values[-2]
            else:
                step_values[-1] = average[group_index]
            pre_group_time = group_time

        # replace the group average with partial average if the whole group is not required
        if not whole_group:
            if cur_step == step or step > 0:
                step_values[0 if step > 0 else -1] = run_average[i]
            elif group_index == 0:
                # this branch is executed only for the first group for backward (few times)
                step_values = deque([run_average[i]] * abs(step))

        splits.append(list(step_values))

    return splits


def round_hour(time: datetime, hours):
    return time.replace(hour=time.hour - time.hour % hours, minute=0, second=0)


def select(df: pd.DataFrame, time_key,
           from_time='00-00-00 00', to_time='99-01-01 00'):
    filter_index = (df[time_key] >= from_time) & (df[time_key] < to_time)
    return df.loc[filter_index, :].reset_index(drop=True)