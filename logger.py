import datetime

def print_with_timestamp(*args):
    timestamp = datetime.datetime.now().strftime("[%d.%m.%Y %H:%M:%S]")
    print(timestamp, *args)
