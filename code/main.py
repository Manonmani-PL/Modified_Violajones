import preprocess.IntegralImage as ii
import AdaBoost as ab
import preprocess.eohsobel as es

if __name__ == "__main__":
    pos_training_path = 'data/face'
    neg_training_path = 'data/not_face'
    pos_testing_path = 'data/face/testface'
    neg_testing_path = 'data/not_face/test_notface'

    num_classifiers = 2
    #feature size
    min_feature_height = 8
    max_feature_height = 10
    min_feature_width = 8
    max_feature_width = 10

    #Loading faces with some basic check
    faces_training = es.load_images(pos_training_path)

    #iterating
    faces_ii_training = list(map(ii.to_integral_image, faces_training))
    print('done faces loaded.')

    non_faces_training = es.load_images(neg_training_path)
    non_faces_ii_training = list(map(ii.to_integral_image, non_faces_training))
    print('done non faces loaded.\n')

    # classifiers are haar like features
    classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)


    print('Loading test faces..')
    faces_testing = es.load_images(pos_testing_path)
    faces_ii_testing = list(map(ii.to_integral_image, faces_testing))
    print('..done faces loaded.\n')
    non_faces_testing = es.load_images(neg_testing_path)
    non_faces_ii_testing = list(map(ii.to_integral_image, non_faces_testing))
    print('..done non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    # Classifies given list of integral images using classifiers,
    correct_faces = sum(es.ensemble_vote_all(faces_ii_testing, classifiers))
    correct_non_faces = len(non_faces_testing) - sum(es.ensemble_vote_all(non_faces_ii_testing, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

    # putting all haar-like features over each other generates a face-like image
    recon = es.reconstruct(classifiers, faces_testing[0].shape)
    count = count+1
    recon.save('dataset/face/reconstruction.png')
