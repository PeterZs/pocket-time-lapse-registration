import time;
import pandas as pd;
from .ADecoratorClass import ADecoratorClass
from functools import wraps
import os


class TimeLog(object):
    def __init__(self, output_directory=None, write_after_each_call=False, current_tag=None):
        self.dataframe = pd.DataFrame(columns=['function_name', 'start_time', 'end_time', 'tag', 'duration', 'notes']);
        self.output_directory = output_directory;
        self.write_after_each_call = write_after_each_call;
        self.current_tag = current_tag;
        self.start_time_string = time.asctime();

    def log_function_time(self, function_name=None, start_time=None, end_time=None, tag=None, notes=None):
        if (tag is None):
            tag = self.current_tag;

        duration = None;
        if(start_time is not None and end_time is not None):
            duration = end_time - start_time;

        self.dataframe.loc[len(self.dataframe.index)] = pd.Series(dict(
            function_name=function_name,
            tag=tag,
            start_time=start_time,
            end_time=end_time,
            duration= duration,
            notes=notes,
        ))
        if (self.write_after_each_call):
            self.write_log()

    def write_log(self, output_path=None):
        if (output_path is None):
            output_path = self.get_default_output_path();
        self.dataframe.to_csv(output_path);

    def get_default_output_path(self, tag=None):
        if (tag is None):
            tag = self.start_time_string;
        output_directory = self.output_directory;
        if (output_directory is None):
            output_directory = '.' + os.sep;
        return os.path.join(output_directory, 'TIMELOG_' + tag + '.csv');


class LogTime(ADecoratorClass):
    is_logging = False;
    time_log = None;
    current_log_start = None;

    @classmethod
    def Start(cls, tag=None, output_dir=None, write_after_each_call=False, notes=None):
        if(not os.path.exists(output_dir)):
            print("CREATING DIRECTORY {}".format(output_dir));
            pparts = os.path.split(output_dir);
            destfolder = pparts[0] + os.sep;
            try:
                os.makedirs(destfolder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        if(cls.is_logging):
            print("STARTED TIME LOG WHILE ONE WAS RUNNING!!!");
            cls.Stop();
        start_time = time.time()
        cls.current_log_start = start_time;
        cls.time_log = TimeLog(
            output_directory=output_dir,
            write_after_each_call=write_after_each_call,
            current_tag = tag,
        )
        cls.time_log.log_function_time(
            function_name="LogTime.Start",
            start_time = start_time,
            end_time = start_time,
            notes=notes,
        )
        cls.is_logging = True;

    @classmethod
    def Stop(cls, output_path=None, notes=None):
        cls.time_log.log_function_time(
            function_name="LogTime.Stop",
            start_time=cls.current_log_start,
            end_time=time.time(),
            notes=notes,
        )
        cls.time_log.write_log(output_path=output_path);
        cls.is_logging = False;

    @classmethod
    def log_function(cls,
                     function_name,
                     start_time,
                     end_time,
                     tag=None,
                     notes=None,
                     ):
        cls.time_log.log_function_time(
            function_name=function_name,
            start_time=start_time,
            end_time=end_time,
            tag=tag,
            notes=notes
        )

    def __init__(self, *args, **decorator_args):
        '''
        When function is declared. PUn
        :param decorator_args:
        '''
        self.decorator_args = decorator_args;


    # def __call__(self, func):
    #     self.ValidateFuncArgs(func);
    #     self.PreDecorate(func);
    #     self.original_func_name = func.__name__;
    #     self.op_name = self.original_func_name;
    #     if (self.nickname is not None):
    #         self.op_name = self.nickname;
    #     self.op_index = self.getOpIndex();
    #     decorated = self.DecorateFunction(func);
    #     self.SetOpInfo(decorated);
    #     self.RegisterOp(decorated);
    #     self.PostDecorate(decorated)
    #     return decorated;

    def decorate_function(self, func):
        function_name = self.original_func_name;
        @wraps(func)
        def decorated(*args, **kwargs):
            start_time = time.time();
            tag = self.__class__.time_log.current_tag;
            rval = func(*args, **kwargs);
            end_time = time.time();
            if(self.__class__.is_logging):
                self.__class__.log_function(
                    function_name=function_name,
                    start_time=start_time,
                    end_time=end_time,
                    tag=tag
                )
            return rval;

        return decorated;
