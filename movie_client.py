import grpc
import movie_pb2
import movie_pb2_grpc

def run():
    print 'input'
    channel = grpc.insecure_channel('[::]:5011')
    stub = movie_pb2_grpc.FindMovieServiceStub(channel)
    print 'input'
    while True:
        string = raw_input()
        query = movie_pb2.FindMovieRequest(query = string)
        movies = stub.FindMovies(query)
        for movie in movies.movies:
            print movie.encode('utf-8')

if __name__ == '__main__':
    run()
