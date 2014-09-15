'''
# Gabe Gaster, 2013
#
# A timer suite of decorators, for time profiling.
#
# - timer is a decorator
#   timer takes in any function
#   keeps track of the time as it executes that function
#   does everything the function does and also reports how long it took 
#
# - eta is a decorator that takes an argument for a method 
#   that will be repeatedly called
# 
#   eta takes the number of expected iterations
#
#   about every minute, eta will report how many iterations 
#   have been performed and the Expected Time to Completion (the E.T.C.)
#
#####################################################################
#
# Examples:
#

from timer import timer

@timer
def timer_test():
    time.sleep(.5)

timer_test()

# prints (to standard error) :
# the function timer_test took 0.500942 seconds

######################################################################

from timer import etc

N_TIMES =20 # num times test is expected to be called
@etc(N_TIMES, update_time=4)
def test(msg):
    time.sleep(1)

if __name__ == "__main__":
    for i in xrange(N_TIMES):
        test(i)

# prints (to standard error) :
  # 0.1 min elapsed, 25.0 % done, ETA:   0.3 min
  # 0.1 min elapsed, 45.0 % done, ETA:   0.3 min
  # 0.2 min elapsed, 65.0 % done, ETA:   0.3 min
  # 0.3 min elapsed, 85.0 % done, ETA:   0.3 min

'''

import time
import sys

# timer is a decorator
# timer takes in any function
# keeps track of the time as it executes that function
# does everything the function does and also reports how long it took
def timer(some_function):
    def timed_function(*args,**kwargs):
        tic = time.time()
        out = some_function(*args,**kwargs)
        tok = time.time()
        template = "the function %s took %f seconds"
        print template%(some_function.__name__,tok - tic)
        return out
    return timed_function    

# eta is a decorator that takes an argument for a method 
# that will be repeatedly called
# 
# eta takes the number of expected iterations
#
# about every minute, eta will report how many iterations have been performed
# and the Expected Time to Completion (the E.T.C.)
class etc(object):
    def __init__(self, total, update_time=60):
        self.total = total
        self.done = 0.
        self.update_time = update_time

    def __call__(self, some_function):
        def wrapped_f(*args,**kwargs):
            now = time.time()
            if not self.done: # only happens when f is first called
                self.begin = self.last_update = now

            self.done += 1.
            if (now - self.last_update) > self.update_time:
                self.last_update = now  # update every minute
                t = now - self.begin
                percent_done = self.done / self.total
                msg = "%5.1f min elapsed, %s %% done, ETA: %5.1f min"
                print >> sys.stderr, msg%(t/60, percent_done*100,
                                          t/percent_done/60)
            return some_function(*args,**kwargs)
        return wrapped_f        

def get_time_str(num_secs):
    minutes = num_secs*1./60
    hours = minutes/60
    days = hours/24
    if num_secs < 60:
        return "%5.1f sec"%num_secs
    if minutes < 60:
        return "%5.1f min"%minutes
    elif hours < 24:
        return "%5.1f hours"%hours
    else:
        return "%5.1f days"%days
        
def show_progress(iterable, update_time=60, length=None):
    try:
        name = iterable.__name__
    except AttributeError:
        name = ""#iterable.__repr__()
    if not hasattr(iterable,"__iter__"):
        raise TypeError("Object %s not iterable"%name)
    if hasattr(iterable,"__len__") and length is None:
        length = iterable.__len__()

    start = last_update = time.time()
    # if the length is unknown, don't estimate completion time,
    # but still show periodic progress updates.    

    for count,thing in enumerate(iterable):
        now = time.time()
        if now - last_update > update_time:
            last_update = now
            t = now - start
            if length:
                percent_done = 1.* count / length
                if percent_done:
                    etc_secs = t/percent_done
                    etc_time = get_time_str(etc_secs)
                else:
                    etc_time = "NA"
                msg = "%s elapsed, %1.3f%% done, ETC: %s"
                msg = msg%(get_time_str(t),percent_done*100,etc_time)
            else:
                msg = "%s elapsed, %s done : %s per iter"
                msg = msg%(get_time_str(t), count, get_time_str(count/t))
            print >> sys.stderr, "%s : %s"%(name,msg)
        yield thing

    print >> sys.stderr, "%s : DONE in %s"%(name,
                                            get_time_str(time.time()-start))

N_TIMES =20 # num times test is expected to be called
@etc(N_TIMES,4)
def test(msg):
    time.sleep(.5)

@timer
def timer_test():
    time.sleep(.5)

if __name__ == "__main__":
    for i in xrange(N_TIMES):
        test(i)
    timer_test()

    for j in show_progress(xrange(20),update_time=4,length=20):
        test(i)
