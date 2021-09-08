import inspect

def foo():

   # get current function name
   print(inspect.stack()[0][3])

   # get caller name)
   #print(inspect.stack()[1][3])


if __name__ == "__main__":
    foo()

