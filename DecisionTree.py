import pandas
from sklearn.tree import DecisionTreeClassifier

# read csv file, if you receive error message while reading it. can add encoding property like below
dataframe = pandas.read_csv("top-rated-movie-genres.csv", encoding='latin-1')

# convert text to number to be used by decision tree
convertToNumber = {'Biography': 0, 'Adventure': 1, 'Comedy': 2, 'Drama': 3, 'Action': 4, 'Animation': 5, 'Crime': 6, 'Drama': 7, 'Mystery': 8}
dataframe['Genre1'] = dataframe['Genre1'].map(convertToNumber)

# Using 'Rating', 'Votes by Male & Female' as X Axis and 'Genre' as Y Axis
features = ['Rating', 'CVotesMale', 'CVotesFemale']
X = dataframe[features]
Y = dataframe['Genre1']

# load data with decision tree algorithm
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

# process to build/export into pdf
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_pdf("my-tree.pdf")

# # process to build/export into png or image
# import pydotplus
# import matplotlib.pyplot as plt
# import matplotlib.image as pltimg
# from sklearn import tree
#
# data = tree.export_graphviz(dtree, out_file=None)
# graph = pydotplus.graph_from_dot_data(data)
#
# # auto export/download into png
# graph.write_png('movie-genre-decisiontree.png')
# img=pltimg.imread('movie-genre-decisiontree.png')
#
# # show image modal in pycharm
# imgplot = plt.imshow(img)
# plt.title("Decision Tree of Movie-Genres")
# plt.show()