# creating module

def square(num):
  return num*num




def cube(num):
  return num**3



def factoriacl(num):
  if(num==1 or num==0): #base fucntion
    return 1
  return num*factoriacl(num-1)


def add(num1,num2):
  return num1+num2

def revesreString(s):
  return s[: : -1]

def pallindrom(s):
  return s==s[: : -1]

def concatString(s,t):
  return s+t

def stringLength(s):
  return len(s)