import datetime


def gen_col_period(row):
    reserve_datetime = datetime.datetime.strptime(row['reserve_datetime'], '%Y-%m-%d %H:%M:%S')
    visit_datetime = datetime.datetime.strptime(row['visit_datetime'], '%Y-%m-%d %H:%M:%S')
    row['reserve_date'] = reserve_datetime.strftime('%Y-%m-%d')
    row['reserve_hour'] = reserve_datetime.strftime('%H')
    row['visit_date'] = visit_datetime.strftime('%Y-%m-%d')
    row['visit_hour'] = visit_datetime.strftime('%H')
    date_delta = visit_datetime - reserve_datetime
    row['period_days'] = date_delta.days
    row['period_hours'] = date_delta.days * 24 + date_delta.seconds / (60 * 60)
    return row


def gen_col_period_dt(row):
    reserve_datetime = row['reserve_datetime']
    visit_datetime = row['visit_datetime']
    row['reserve_date'] = reserve_datetime.strftime('%Y-%m-%d')
    row['reserve_hour'] = reserve_datetime.strftime('%H')
    row['visit_date'] = visit_datetime.strftime('%Y-%m-%d')
    row['visit_hour'] = visit_datetime.strftime('%H')
    date_delta = visit_datetime - reserve_datetime
    row['period_days'] = date_delta.days
    row['period_hours'] = date_delta.days * 24 + date_delta.seconds / (60 * 60)
    return row
