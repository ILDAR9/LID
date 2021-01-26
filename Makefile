STORE=/Users/a18180846/projects/data
#SETTINGS=./experiments/sbcnn32mel256hop.yml
SETTINGS=./experiments/sbcnn32mel512hop.yml
TRAIN_DATA=./data/features_32mel512.csv

preprocess_ru:
	source env/bin/activate && \
	 python -m lid.preprocessing --jobs 1 --lang ru --store ${STORE} --settings ${SETTINGS}
preprocess_de:
	source env/bin/activate && \
	 python -m lid.preprocessing --jobs 1 --lang de --store ${STORE} --settings ${SETTINGS}

preprocess_en:
	source env/bin/activate && \
	 python -m lid.preprocessing --jobs 1 --lang en --store ${STORE} --settings ${SETTINGS}

featurize_ru:
	source env/bin/activate && \
	 python -m lid.featurization --jobs 1 --lang ru --store ${STORE} --settings ${SETTINGS}

featurize_de:
	source env/bin/activate && \
	 python -m lid.featurization --jobs 1 --lang de --store ${STORE} --settings ${SETTINGS}

featurize_en:
	source env/bin/activate && \
	 python -m lid.featurization --jobs 1 --lang en --store ${STORE} --settings ${SETTINGS}

train:
	source env/bin/activate && \
	 python -m lid.train --data ${TRAIN_DATA} --store ${STORE} --settings ${SETTINGS} --data ${TRAIN_DATA}