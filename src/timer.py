
import time

class Clock:
    def __init__(self, start_tick = True):
        self.pre_time = 0
        if start_tick:
            self.start()

    def start(self):
        self.pre_time = time.time()

    def stop(self):
        self.cur_time = time.time()
        print('time {} elapsed!'.format(self.cur_time - self.pre_time))
        self.pre_time = self.cur_time 