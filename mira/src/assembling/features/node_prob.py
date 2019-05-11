import ujson, numpy as np, sys
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

workdir = Path(sys.argv[1])

with open(workdir / "sm_node_prob_tt8tty1.json", "r") as f:
    data = ujson.load(f)
    x_train = np.asarray(data["matrix"])
    y_train = [int(x) for x in data["labels"]]
    is_only_one_class = all(y_train)

    if is_only_one_class:
        # this should be at a starter phase when we don't have any data but use ground-truth to build
        x_train = np.vstack([x_train, [0, 0]])
        y_train.append(0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    classifier = LogisticRegression(fit_intercept=True)
    classifier.fit(x_train, y_train)

    if is_only_one_class:
        result = classifier.predict_proba(x_train)[:-1, 1]
    else:
        result = classifier.predict_proba(x_train)[:, 1]

    # TODO: uncomment to report performance
    # print(classification_report(y_train, classifier.predict(x_train)))

    output = {
        "scaler": {
            "mean": scaler.mean_,
            "scale": scaler.scale_
        },
        "classifier": {
            "intercept": classifier.intercept_[0],
            "coef": classifier.coef_[0]
        },
        "result": result
    }

    with open(workdir / "sm_node_prob_tt8tty1.output.json", "w") as g:
        ujson.dump(output, g, indent=4)

