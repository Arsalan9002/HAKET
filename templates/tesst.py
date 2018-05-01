
import sys

def output(argument):
    if argument == 0:
        return 0
    elif argument == 1:
        return 2
    else:
        return (output(argument-1)+output(argument-2))


print(output(5))