import detectors


def test_create_detector_docstring():
    ds = detectors.list_detectors()
    assert detectors.create_detector.__doc__ is not None
    for d in ds:
        print(d, d in detectors.create_detector.__doc__)
    assert all([d in detectors.create_detector.__doc__ for d in ds])


def test_create_pipeline_docstring():
    pipelines = detectors.list_pipelines()
    assert detectors.create_pipeline.__doc__ is not None
    for p in pipelines:
        print(p, p in detectors.create_pipeline.__doc__)
    assert all([p in detectors.create_pipeline.__doc__ for p in pipelines])


def test_create_dataset_docstring():
    datasets = detectors.list_datasets()
    assert detectors.create_dataset.__doc__ is not None
    for d in datasets:
        print(d, d in detectors.create_dataset.__doc__)
    assert all([d in detectors.create_dataset.__doc__ for d in datasets])
