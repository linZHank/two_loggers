class bcolors:
    """ For the purpose of print in terminal with colors """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
# bonus functions
def bonusWallDividedNumsteps(bonus_wall, num_steps): return bonus_wall/num_steps # bonus for every time step
def weightedD0(weight,d0): return weight*d0 # bonus for initial distance
def d0MinusD(d0,d,num_stes): return (d0-d)/num_steps # bonus of approaching the exit
def zero(x,y,z=0): return 0

