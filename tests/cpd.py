from detectors.methods.cpd.pelt import PeltL2
from detectors.methods.cpd.utils import pw_constant


def test_pelt_l2_cpd_detector():
    detector = PeltL2()
    signal, bkps = pw_constant(n_samples=11000)
    detector.fit(signal.reshape(-1))
    bkps_detected = detector.predict(pen=1)
    print(bkps_detected)
    print(bkps)


if __name__ == "__main__":
    test_pelt_l2_cpd_detector()
