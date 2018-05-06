import json
import pandas as pd
import const
import settings

config = settings.config[const.DEFAULT]

# load json file into a dictionary
json_file = open(config[const.LEADERBOARD_JSON])
json_data = json.loads(json_file.read())

valid_month = 5  # month to be considered for scores
start_day = 1  # day to start averaging scores
end_day = 30  # day to end averaging scores

# decode A2018_4_11 into month: 4, and day: 11
# and find the last day
dates = json_data[0]['date']
date_map = dict()
last_day = -1
for date in dates:
    valid_month = int(date.split('_')[1])
    day = int(date.split('_')[2])
    date_map[date] = {'month': valid_month, 'day': day}
    if day > last_day:
        last_day = day

team_count = len(json_data) - 1
score = dict()
count = dict()
for t in range(1, team_count + 1):
    data = json_data[t]
    name = data['team_name']
    score[name] = 0
    count[name] = 0
    for date_code in data:
        if date_code not in dates:
            continue
        month = int(date_map[date_code]['month'])
        day = int(date_map[date_code]['day'])
        value = float(data[date_code])
        if start_day <= day < end_day and month == valid_month and value < 2:
            score[name] += value
            count[name] += 1
    score[name] = score[name] / count[name] if count[name] > 0 else 2

team_rank = list()
team_name = list()
team_score = list()
team_count = list()
rank = 1
for team, score in sorted(score.items(), key=lambda item: (item[1], item[0])):
    team_rank.append(rank)
    team_name.append(team)
    team_score.append(score)
    team_count.append(count[team])
    rank = rank + 1
    print("%s\t%s\t%.3f\t%d" % (rank, team, score, count[team]))

score_title = 'score %d [%d, %d]' % (valid_month, start_day, last_day)

df = pd.DataFrame(data={
    'rank': team_rank,
    'name': team_name,
    score_title: team_score,
    'count': team_count,
}, columns=['rank', 'name', score_title, 'count'])

print('Done!')
