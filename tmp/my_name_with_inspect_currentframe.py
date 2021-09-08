import inspect

def foo():

   # more fast way since do not retrieve a lias  of all stack frames as inspect.stack() doess
   # get current function name
   print(inspect.currentframe().f_code.co_name)

   # get caller name)
   #print(inspect.currentframe().f_back.f_code.co_name)


if __name__ == "__main__":
    foo()

