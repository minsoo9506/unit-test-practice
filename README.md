- test 위치에서 `pytest`를 명령하면 test에 있는 모든 test file을 test 진행한다.
- 그럼 아래와 같은 결과가 나온다. (일부러 error 만듬)
```
(base) C:\Users\ghktj\Desktop\unit-test-practice\test>pytest
======================================================== test session starts =========================================================
platform win32 -- Python 3.8.3, pytest-5.4.3, py-1.9.0, pluggy-0.13.1
rootdir: C:\Users\ghktj\Desktop\unit-test-practice\test
collected 4 items

preprocess\test_standardize.py ..                                                                                               [ 50%]
train\test_LinearRegression.py .F                                                                                               [100%]

============================================================== FAILURES ==============================================================
_____________________________________________________ TestTrain.test_train_pred ______________________________________________________

self = <test.train.test_LinearRegression.TestTrain object at 0x000002984B136040>

    def test_train_pred(self):
        X, y, w =generate_data(n_samples=100, n_features=2, bias=1.0)
        test_X = np.array([[1.0, 1.0]])
        test_y = w * test_X + 1.0
        model = train_LinearRegression(X, y)
        pred = model.predict(test_X)
>       assert test_y == pytest.approx(pred), 'Wrong prediction!'
E       AssertionError: Wrong prediction!
E       assert array([[30.21475268, 97.19363785]]) == approx([126.40839053397774 ± 1.3e-04])
E        +  where approx([126.40839053397774 ± 1.3e-04]) = <function approx at 0x0000029847779820>(array([126.40839053]))
E        +    where <function approx at 0x0000029847779820> = pytest.approx

train\test_LinearRegression.py:23: AssertionError
====================================================== short test summary info ======================================================= 
FAILED train/test_LinearRegression.py::TestTrain::test_train_pred - AssertionError: Wrong prediction!
==================================================== 1 failed, 3 passed in 1.20s ===================================================== 
```