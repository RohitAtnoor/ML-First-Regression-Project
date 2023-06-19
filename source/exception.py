import sys  # System library for Custom exception handeling librar .
from source.logger import logging  # importing the logging function from logger file from source folder.

# Custome defined error message function for displaying the error message.
def error_message_detail(error,error_detail:sys):
    # error is the error message which we get, error_detail is the error message raised by the system.
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    # Exception is the parent class for handeling the exception.
    # CustomException is the child class which inherit the parent Exception class.
    
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message    

#  Example.

"""
if __name__=="__main__":
    logging.info("Logging has started")
    try:
        a=1/0
    except Exception as e:
        logging.info('Division by zero') 
        raise CustomException(e,sys)

"""