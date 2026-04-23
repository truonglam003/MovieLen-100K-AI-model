RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
LINKS_PATH = "data/links.csv"

TMDB_API_KEY = "1e529565a6baf86e9bd50b4ae6299437"  # Điền API Key của TMDB vào đây hoặc nhập trên giao diện web

TOP_N = 10

MIN_USER_RATINGS = 5
MIN_MOVIE_RATINGS = 5

N_COMPONENTS = 15
RANDOM_STATE = 42

NEGATIVE_SAMPLE_SIZE = 10
POPULARITY_MIN_RATINGS = 10

RUN_EVALUATION = False              # here
MAX_EVAL_USERS = 2000

# giới hạn số user khi evaluate để đỡ chậm
# MAX_EVAL_USERS = 500

'''
Khi cần lấy số liệu cho báo cáo
RUN_EVALUATION = True
'''