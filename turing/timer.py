import time

class ThroughputTimer(object):
    def __init__(self, name = None, batch_size=1, num_workers=1, start_step=2):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.count = 0
        self.total_elapsed_time = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.start_step = start_step
        self.name = name

    def start(self, cond=True):
        if cond:
            self.start_time = time.time()
            self.started = True

    def stop(self, cond=True):
        if cond and self.started:
            self.end_time = time.time()
            self.started = False
            self.count += 1
            if self.count >= self.start_step:
                self.total_elapsed_time += self.end_time - self.start_time
        elif cond and not self.started:
            print("Cannot stop timer without starting ")
            exit(0)

    def avg_samples_per_sec(self):
        if self.count > 2:
            samples_per_step = self.batch_size * self.num_workers
            avg_time_per_step = self.total_elapsed_time / (self.count-2.0)
            # training samples per second
            return samples_per_step / avg_time_per_step
        return -999

    def avg_steps_per_sec(self):
        if self.count > 2:
            return 1 / (self.total_elapsed_time / (self.count-2.0))
        return -999


    def print_elapsed_time(self, num_ops=None):
        if self.count > 2 and self.count % 1000  == 0:
            elapsed_time = self.total_elapsed_time / (self.count-2.0)
            if num_ops == None:
                print(self.name, " forward pass execution time: ", elapsed_time)
            else:
                print(self.name, " forward pass execution time: ", elapsed_time, " TFlops : ", num_ops/(elapsed_time * 1000000000000))
