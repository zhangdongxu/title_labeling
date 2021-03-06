from concurrent import futures
import time
import sys
import grpc
import movie_pb2
import movie_pb2_grpc
import argparse
from descriptor import Descriptor

parser = argparse.ArgumentParser()
parser.add_argument("--model",help="the path of the model to save or load",\
        required=True)
parser.add_argument("--address", help="the ip and port this service want to listen", default="[::]:5011")
parser.add_argument("--topk", help="top k", type=int, default=50)
args = parser.parse_args()
descriptor = Descriptor()
descriptor.load_model(args.model, "max")

class movieServicer(movie_pb2_grpc.FindMovieServiceServicer):
    def FindMovies(self, request, context):
        query = request.query
        print(time.strftime('%Y-%m-%d/%H:%M:%S', time.localtime(time.time())) + '\t' + query)
        sys.stdout.flush()
        ngram_desc = descriptor.match_desc_max(query)
        titles = descriptor.rank_titles(ngram_desc, args.topk)
        movies = [title for title, _, _ in titles]
        return movie_pb2.FindMovieReply(movies=movies)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    movie_pb2.add_FindMovieServiceServicer_to_server(movieServicer(), server)
    server.add_insecure_port(args.address)
    server.start()
    print("service started on " + args.address)
    sys.stdout.flush()
    while True:
        time.sleep(1)

if __name__ == '__main__':
    serve()
