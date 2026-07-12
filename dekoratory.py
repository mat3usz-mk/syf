import logging

def add_print(fun: function)-> function:
    logging.basicConfig(filename='{}.log'.format(fun.__name__), level=logging.INFO)
    def wrapper(*args,**kwargs):

        print("wyniki działania liczb {}".format(fun.__name__))
        print('odpalono z liczbami: args: {} i kwargs: {} '.format(args, kwargs))
        logging.info('prazetowrzono funkcje z args: {} i kwargs: {}'.format(args, kwargs))
        return fun(*args,*kwargs)
    return wrapper


@add_print
def sigiemka(n1:int, n2:int) ->int:
    return print(n1+n2)


sigiemka(10,20)