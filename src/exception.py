import sys
import logging

def get_detailed_error_message(error, error_detail:sys):
        """
        Formats the error message with traceback information.
        """
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occurred in file [{0}] at line [{1}] error_message [{2}]".format(
                file_name,exc_tb.tb_lineno,str(error)
        )

        return error_message

class CustomException(Exception):
    """
    Custom exception class for handling project-specific errors.
    """
    def __init__(self, error_message, error_detail:sys):
        """
        Initialize with a detailed error message and traceback details.
        """
        super().__init__(error_message)
        self.error_message = get_detailed_error_message(error_message, error_detail)

    def __str__(self):
        return self.error_message

    
