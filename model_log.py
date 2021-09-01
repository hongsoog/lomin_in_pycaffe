import logging

logging.basicConfig()
logger = logging.getLogger()
# Debug < Info < Warning < Error < Critical
#    10     20      30       40       50
# logging level INFO includes info, Warning, Error, Critical
# hence exclude DEBUG
#logger.setLevel(logging.INFO)


# logging level DEBUG
# include DEBUG and above
logger.setLevel(logging.CRITICAL)