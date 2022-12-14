import os


def create_logger(log_filename, display=True, log_append = False):

    if log_append is False:
        f = open(log_filename, 'w')
    else:
        f = open(log_filename, 'a')

    counter = [0]
    # this function will still have access to f after create_logger terminates

    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 1 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close
