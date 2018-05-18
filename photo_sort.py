import os
import json
import shutil

# Open the json file so to get the labels
os.chdir("yelp_photos")
photo_file = open('photos.json', 'r')

os.chdir("photos")

photos = []

for line in photo_file:
    # print(line)
    photo = json.loads(line)
    photos.append(photo)

photo_file.close()

foodCount = 132354
drinkCount = 6620

drink_training_number = drinkCount * 0.8
food_training_number = foodCount * 0.8

drink_loading_count = 0
food_loading_count = 0

print(drink_training_number, food_training_number)
for photo in photos:
    if photo['label'] == 'food' and food_loading_count < food_training_number:
        ''' shutil.move("/Users/chloechen/project/yelp_photos/photos/"+photo['photo_id']+".jpg",
            "/Users/chloechen/project/yelp_photos/photos/food/"+photo['photo_id']+".jpg")
         print("moving " + photo['photo_id']) '''
        # foodCount += 1

        shutil.move("/Users/chloechen/project/yelp_photos/photos/food/" + photo['photo_id'] + ".jpg",
                    "/Users/chloechen/project/yelp_photos/photos/food/training/" + photo['photo_id'] + ".jpg")
        print("moving " + photo['photo_id'])

        food_loading_count += 1


    elif photo['label'] == 'drink' and drink_loading_count < drink_training_number:
        ''' shutil.move("/Users/chloechen/project/yelp_photos/photos/" + photo['photo_id'] + ".jpg",
            "/Users/chloechen/project/yelp_photos/photos/drink/" + photo['photo_id'] + ".jpg")
        print("moving " + photo['photo_id']) '''
        # drinkCount += 1

        shutil.move("/Users/chloechen/project/yelp_photos/photos/drink/" + photo['photo_id'] + ".jpg",
                    "/Users/chloechen/project/yelp_photos/photos/food/training/" + photo['photo_id'] + ".jpg")
        print("moving " + photo['photo_id'])
        drink_loading_count += 1

    else:
        pass

