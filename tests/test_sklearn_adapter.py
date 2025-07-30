from gen_surv.sklearn_adapter import GenSurvDataGenerator


def test_sklearn_generator_dataframe():
    gen = GenSurvDataGenerator(
        "cphm",
        n=4,
        beta=0.2,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=1.0,
    )
    df = gen.fit_transform()
    assert len(df) == 4
    assert {"time", "status"}.issubset(df.columns)


def test_sklearn_generator_dict():
    gen = GenSurvDataGenerator(
        "cphm",
        return_type="dict",
        n=3,
        beta=0.5,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=1.0,
    )
    data = gen.transform()
    assert isinstance(data, dict)
    assert set(data.keys()) >= {"time", "status"}
    assert len(data["time"]) == 3
