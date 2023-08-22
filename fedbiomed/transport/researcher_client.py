 

import asyncio
import grpc
import queue
import threading
import sys 
import signal 
from enum import Enum

from typing import Callable, Dict

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.proto.researcher_pb2 import TaskRequest, FeedBackMessage
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer

import uuid
import time

import statistics

class GRPCStop(Exception):
    """Stop exception for gRPC client"""
    pass 

class GRPCTimeout(Exception):
    """gRPC timeout error"""
    pass 

class ClientStop(Exception):
    pass 

class GRPCStreamingKeepAliveExceed(Exception):
    """"Task reader keep alive error"""
    pass

logger.setLevel("DEBUG")
NODE_ID = str(uuid.uuid4())




DEFAULT_ADDRESS = "localhost:50051"
STREAMING_MAX_KEEP_ALIVE_SECONDS = 60 


SHUTDOWN_EVENT =  threading.Event()
SHUTDOWN_FORCE_EVENT = threading.Event()
REQUEST_DONE = threading.Event()



def create_channel(
    address: str = DEFAULT_ADDRESS ,
    certificate: str = None
) -> grpc.Channel :
    """ Create gRPC channel 
    
    Args: 
        ip: 
        port:
        certificate:
    
    Returns: 
        gRPC connection channel
    """
    channel_options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]

    if certificate is None: 
        channel = grpc.aio.insecure_channel(address, options=channel_options)
    else:
        # TODO: Create secure channel
        pass
    
    # TODO: add callback fro connection state

    return channel

class TaskReturnCode(Enum):
    TRC_UNKNOWN = 0
    TRC_CANCELLED = 1
    TRC_ERROR_UNAVAIL = 2
    TRC_ERROR_TIMEOUT = 3


async def task_reader_unary(
        stub, 
        node: str,
        callback: Callable = lambda x: x,
        debug: bool = False
         
) -> TaskReturnCode:
    """Task reader as unary RPC
    
    This methods send unary RPC to gRPC server (researcher) and get tasks 
    in a stream. Stream is used in order to receive larger messages (more than 4MB)
    After a task received it sends another task request immediately. 

    Args: 
        stub: gRPC stub to execute RPCs. 
        node: Node id 
        callback: Callback function to execute each time a task received
    """
    
    # def request_done(x):
    #     REQUEST_DONE.set()

    #while not SHUTDOWN_EVENT.is_set():


    try:
        trc = TaskReturnCode.TRC_UNKNOWN
        while True:
            #request_iterator = stub.GetTaskUnary(
            #    TaskRequest(node=f"{node}")
            #)
            request_iterator_future = stub.GetTaskUnary(
                TaskRequest(node=f"{node}")
            )
            #request_iterator.cancel()
            # request_iterator.add_done_callback(request_done)
            if debug: print("task_reader_unary: launching generator")
            # Prepare reply
            reply = bytes()
            async for answer in request_iterator_future:

                try:
                    # print(f"print {threading.current_thread()} {asyncio.current_task()}")
                    reply += answer.bytes_
                    if answer.size != answer.iteration:
                        continue
                    else:
                        # Execute callback
                        callback(Serializer.loads(reply))
                        # Reset reply
                        reply = bytes()
                # TODO: does this case occur ? not seen yet ...
                except Exception:
                    logger.error("gRPC generator: exception")
                finally:
                    if debug: print("gRPC generator: finally")

    except grpc.aio.AioRpcError as exp:
        if debug: print(f"task_reader_unary: grpc.aio exception {exp.code()}")
        if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            trc = TaskReturnCode.TRC_ERROR_TIMEOUT
        elif exp.code() == grpc.StatusCode.UNAVAILABLE:
            trc = TaskReturnCode.TRC_ERROR_UNAVAIL
        else:
            # TODO: does this case occur ? not seen yet.
            #raise Exception("Request streaming stopped ") from exp
            pass

    except asyncio.CancelledError as e:
        trc = TaskReturnCode.TRC_CANCELLED
        if debug: print(f'task_reader_unary: exception cancel: {e}')
        request_iterator_future.cancel()
        # probably not needed here
        #raise asyncio.CancelledError
    except Exception as e:
        if debug: print(f"task_reader_unary: exception generic: {e}")
        request_iterator_future.cancel()
    finally:
        if debug: print('task_reader_unary: finally')
        return trc

#async def task_reader(
#        stub, 
#        node: str, 
#        callback: Callable = lambda x: x
#        ) -> None:
#        """Low-level task reader implementation 
#
#        This methods launches two coroutine asynchronously:
#            - Task 1 : Send request stream to researcher to get tasks 
#            - Task 2 : Iterates through response to retrieve tasks  
#
#        Task request iterator stops until a task received from the researcher server. This is 
#        managed using asyncio.Condition. Where the condition is "do not ask for new task until receive one"
#
#        Once a task is received reader fires the callback function
#
#        Args: 
#            node: The ID of the node that requests for tasks
#            callback: Callback function that takes a single task as an arguments
#
#        Raises:
#            GRPCTimeout: Once the timeout is exceeded reader raises time out exception and 
#                closes the streaming.
#        """  
#        event = asyncio.Event()
#
#        async def request_tasks(event):
#            """Send request stream to researcher server"""
#            async def request_():
#                # Send starting request without relying on a condition
#                count  = 1
#                logger.info("Sending first request after creating a new stream connection")
#                state.n = time.time()
#                yield TaskRequest(node=f"{count}---------{node}")
#                                
#                while True:
#                    # Wait before getting answer from previous request
#                    await event.wait()
#                    event.clear()
#
#                    #logger.info(f"Sending another request ---- within the stream")   
#                    state.n = time.time()  
#                    yield TaskRequest(node=f"{count}---------{node}")
#                    
#
#
#            # Call request iterator withing GetTask stub
#            state.task_iterator = stub.GetTask(
#                        request_()
#                    )
#
#        async def receive_tasks(event):
#            """Receives tasks form researcher server """
#            async for answer in state.task_iterator:
#                
#                reply += answer.bytes_
#                # print(f"{answer.size}, {answer.iteration}")
#                if answer.size != answer.iteration:
#                    continue
#                else:
#                    event.set()
#                    callback(Serializer.loads(reply))
#                    reply = bytes()
#                    
#                
#        # Shared state between two coroutines
#        state = type('', (), {})()
#
#        await asyncio.gather(
#            request_tasks(event), 
#            receive_tasks(event),
#            )


def dummy_callback(x):
    print(f"In callback {x}")


class ResearcherClient:
    """gRPC researcher component client 
    
    Attributes: 


    """
    def __init__(
            self,
            handler = None,
            certificate: str = None,
            debug: bool = True
        ):


        # TODO: implement TLS 
        if certificate is not None:
            # TODO: create channel as secure channel 
            pass 
        
        self._client_registered = False
        self._stop_event = threading.Event()
        #self._client_thread = threading.Thread()

        self._debug = debug

    #async def connection(self):
    #    """Create long-lived connection with researcher server"""
    #    
    #    self._feedback_channel = create_channel(certificate=None)
    #    self._log_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)
#
    #    self._task_channel = create_channel(certificate=None)
    #    self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)
#
    #    logger.info("Waiting for researcher server...")
#
    #    # Starts loop to ask for
    #    try:
    #        await self.get_tasks()   
    #    # TODO: never reached as we handle in `get_tasks`
    #    #
    #    #except Exception as e:
    #    #    print("connection: exception")
    #    finally:
    #        print("connection: finally")

    async def get_tasks(self, debug: bool):

        #self._feedback_channel = create_channel(certificate=None)
        #self._log_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)

        self._task_channel = create_channel(certificate=None)
        self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)

        logger.info("Waiting for researcher server...")

        while True:
            logger.info("Sending new task request")
            try:
                #task = asyncio.create_task(task_reader_unary(stub= self._stub, node=NODE_ID, callback= lambda x: x))
                task = asyncio.create_task(task_reader_unary(stub=self._stub, node=NODE_ID, callback=dummy_callback, debug=debug))
                #print(f"get_tasks: create {asyncio.current_task()} \n{asyncio.all_tasks()}")
                while not task.done():
                    if debug: print("get_task: waiting for task to complete")
                    # note: when receiving a ClientStop, no exception is raised here but execute the `finally`
                    await asyncio.wait({task}, timeout=2)
                if debug: print("get_task: task completed")

            #except GRPCStop:
            #    # Break gRPC while loop
            #    logger.info("Shutting down node gRPC client... ")
            #    break
            #except grpc.aio.AioRpcError as exp:
            #    print(f"get_tasks: grpc.aio exception {exp.code()}")
            #    if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            #        logger.debug("Stream TIMEOUT Error")
            #        await asyncio.sleep(2)
            #
            #    elif exp.code() == grpc.StatusCode.UNAVAILABLE:
            #        logger.debug("Researcher server is not available, will retry connect in 2 seconds")
            #        await asyncio.sleep(2)
            #    else:
            #        raise Exception("Request streaming stopped ") from exp

            # INFO: seems to be never reached - `wait` always complete even if task is interrupted
            #
            # except Exception:
            #    print("get_tasks: exception")
            finally:
                if debug: print("get_tasks: finally")
                #print(f"get_tasks: finally {asyncio.current_task()} \n{asyncio.all_tasks()}")
                if not task.done():
                    if debug: print("get_tasks: cancel")
                    task.cancel()
                while not task.done():
                    await asyncio.sleep(0.1)
                    if debug: print(f"get_tasks: finally cancelling: cancelled {task.cancelled()}")
                
                # we can now read the return code (need to catch `asyncio.CancelledError` if not waiting for `task.done()`)
                res = task.result()
                match res:
                    case TaskReturnCode.TRC_UNKNOWN:
                        logger.error("get_tasks: ERROR bad return code, exiting")
                        break
                    case TaskReturnCode.TRC_CANCELLED:
                        logger.info("get_tasks: cancelled by user, exiting")
                        break
                    case TaskReturnCode.TRC_ERROR_UNAVAIL:
                        logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                        await asyncio.sleep(2)
                    case TaskReturnCode.TRC_ERROR_TIMEOUT:
                        logger.debug("Stream TIMEOUT Error")
                        await asyncio.sleep(2)

    async def send_log(self, log):

        _ = await self._log_tub.Feedback(Log(log=log))


    def start(self):
        """Starts researcher gRPC client"""
        # Runs gRPC async client 
        def run(event):
            try: 
                asyncio.run(
                        #self.connection(), debug=False
                        self.get_tasks(debug=self._debug), debug=False
                    )

            # note: needed to catch this exception
            except ClientStop:
                if self._debug: print("Run: caught user stop exception")
            # TODO: never reached, suppress ?
            #
            except Exception as e:
                if self._debug: print(f"Run: caught exception: {e.__class__.__name__}")
            finally:
                if self._debug: print("Run: finally")

        
        self._t = threading.Thread(target=run, args=(self._stop_event,))
        self._t.start()
        if self._debug: print("start: completed")


    def stop(self, force: bool = False):
        """Stop gently running asyncio loop and its thread"""

        #SHUTDOWN_EVENT.set()
        #SHUTDOWN_FORCE_EVENT.set()

        import ctypes
        stopped_count = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._t.ident),
                                                                   ctypes.py_object(ClientStop))
        if stopped_count != 1:
            logger.error("stop: could not deliver exception to thread")
        self._t.join()

if __name__ == '__main__':
    
    rc= ResearcherClient(debug=False)
    rc.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Node cancel by keyboard interrupt")
        try:
            rc.stop()
        except KeyboardInterrupt:
            print("Already canceling by keyboard interrupt")
