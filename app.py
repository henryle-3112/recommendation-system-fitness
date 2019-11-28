import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:12345@localhost/greenwich_fitness"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class CoachRate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rate = db.Column(db.Integer, primary_key=True)
    coach_id = db.Column(db.Integer, unique=False)
    user_profile_id = db.Column(db.Integer, unique=False)

    def __init__(self, id, rate, coach_id, user_profile_id):
        self.id = id
        self.rate = rate
        self.coach_id = coach_id
        self.user_profile_id = user_profile_id

    def to_json(self):
        return {
            "id": self.id,
            "rate": self.rate,
            "coach_id": self.coach_id,
            "user_profile_id": self.user_profile_id
        }


class Coach(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_profile_id = db.Column(db.Integer, unique=False)
    about = db.Column(db.String(255), unique=False)
    status = db.Column(db.Integer, unique=False)
    rating_average = db.Column(db.Integer, unique=False)
    number_of_memberships = db.Column(db.Integer, unique=False)

    def __init__(self, id, user_profile_id, about, status, rating_average, number_of_memberships):
        self.id = id
        self.user_profile_id = user_profile_id
        self.about = about
        self.status = status
        self.rating_average = rating_average
        self.number_of_memberships = number_of_memberships

    def to_json(self):
        return {
            "id": self.id,
            "user_profile_id": self.user_profile_id,
            "about": self.about,
            "status": self.status,
            "rating_averge": self.rating_average,
            "number_of_memberships": self.number_of_memberships
        }


class CF(object):
    # """docstring for CF"""
    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF  # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k  # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0]))
        self.n_items = int(np.max(self.Y_data[:, 1]))

    def add(self, new_data):
        # """
        # Update Y_data matrix when new ratings come.
        # For simplicity, suppose that there is no new user or item.
        # """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

    def normalize_Y(self):
        users = self.Y_data[:, 0]  # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users + 1,))
        for n in range(1, self.n_users + 1):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1]
            # and the corresponding ratings
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important
        # for both memory and computing efficiency. For example, if #user = 1M,
        # #item = 100k, then shape of the rating matrix would be (100k, 1M),
        # you may not have enough memory to store this. Then, instead, we store
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])),
                                      (self.n_items + 1, self.n_users + 1))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        # """
        # Normalize data and calculate similarity matrix again (after
        # some few ratings added)
        # """
        self.normalize_Y()
        self.similarity()

    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized=1):
        # """
        # predict the rating of user u for item i (normalized)
        # if you need the un
        # """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized=1):
        # """
        # predict the rating of user u for item i (normalized)
        # if you need the un
        # """
        if self.uuCF:
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    def recommend(self, u, normalized=1):
        # """
        # Determine all items should be recommended for user u. (uuCF =1)
        # or all users who might have interest on item u (uuCF = 0)
        # The decision is made based on all i such that:
        # self.pred(u, i) > 0. Suppose we are considering items which
        # have not been rated by u yet.
        # """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(1, self.n_items + 1):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)

        return recommended_items

    def get_recommendation(self, user_id):
        # """
        # print all items which should be recommended for each user
        # """
        # print("Recommendation: ")
        # for u in range(1, self.n_users + 1):
        #     recommended_items = self.recommend(u)
        #     if self.uuCF:
        #         print("Recommend item(s): %s to user %s" %
        #               (recommended_items, u))
        #     else:
        #         print("Recommend item %s to user(s): %s" %
        #               (u, recommended_items))
        recommended_items = self.recommend(user_id)
        if self.uuCF:
            return recommended_items
        return []


# write api

@app.route("/recommend-coach/<user_id>")
def recommend_coach_by_user_id(user_id):
    # check user existed in rating table or not
    selected_user = CoachRate.query.filter_by(user_profile_id=user_id).first()
    if selected_user == None:
        return jsonify([])
    # get all data from rating table in the database
    rating_list = CoachRate.query.all()
    # create rating json list
    Y_data = []
    for each_rating in rating_list:
        # convert data from rating table to matrix
        Y_data.append([int(each_rating.user_profile_id), int(each_rating.coach_id), int(each_rating.rate)])
    Y_data = np.array(Y_data)
    rs = CF(Y_data, k=2, uuCF=1)
    rs.fit()
    recommended_items_id = rs.get_recommendation(int(user_id))
    recommended_items_json_list = []
    if len(recommended_items_id) == 0:
        return jsonify(recommended_items_json_list)
    for i in range(len(recommended_items_id)):
        each_coach = Coach.query.filter_by(
            id=recommended_items_id[i]).first()
        recommended_items_json_list.append(each_coach.to_json())
        print("coach id: %d" % each_coach.id)
    return jsonify(recommended_items_json_list)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
