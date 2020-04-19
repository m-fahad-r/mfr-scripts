# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:29:03 2020

@author: fahad
"""

def main():

    name = input("Please provide your name: ")
    print("")
    print("Hello agent", name)
    print("")
    print("The following script will provide the sum of digits for a number of your choice")
    
    n = int(input("enter the number: "))
    solution = sum_digit(n)
    
    print ("Sum of all the digits is", solution)
    print("")
    input("press enter to end script")
    print("good bye", name)

def sum_digit(n):
    
    """ recursive function, which calls the function within itself to
        complete its objective"""  
    """float(input("enter number: "))"""
    
    if n < 10:
    
        return n
    
    else:
        all_but_last = n // 10 
        last = n % 10
        
        return sum_digit(all_but_last) + last
    
main()