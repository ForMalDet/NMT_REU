#!/usr/bin/python3
import sys

def clear():
    """Clear screen, return cursor to top left"""
    sys.stdout.write('\033[2J')
    sys.stdout.write('\033[H')
    sys.stdout.flush()

def prompt(msg="Select an option:", options=[]):
    while True:
        print("\n" + msg)
        count = 0
        for option in options:
            print("\t{}) {}".format(count, option))
            count += 1
        res = input(" > ")
        if  len(options) == 0 or int(res) < len(options):
            return res
        else:
            print("Please select a number between 0 and {}".format(len(options)-1))

