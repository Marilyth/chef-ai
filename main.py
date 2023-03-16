from Data.data import get_ingredient_lists, encode_ingredient_lists, get_ingredient_counts, crawl_food_com
import Models.MLP

if __name__ == "__main__":
    crawl_food_com(0, 500)
    #Models.MLP.create_mlp()