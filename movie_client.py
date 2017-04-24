import grpc
import movie_pb2
import movie_pb2_grpc
import time

def run():
    channel = grpc.insecure_channel('[::]:5011')
    stub = movie_pb2_grpc.FindMovieServiceStub(channel)
    print('Please input')
    while True:
        string = input()
        print('------------------')
        t1 = time.perf_counter()
        query = movie_pb2.FindMovieRequest(query = string)
        movies = stub.FindMovies(query)
        t2 = time.perf_counter()
        print("response time:" + str(t2 - t1))
        for movie in movies.movies:
            print(movie)
        print('------------------')

if __name__ == '__main__':
    run()
