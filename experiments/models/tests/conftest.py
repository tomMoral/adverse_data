

def pytest_generate_tests(metafunc):
    # Preprocess the collected test to generate the correct tests to run
    if hasattr(metafunc.function, "_repeat"):
        count = metafunc.function._repeat

        # We're going to duplicate these tests by parametrizing them,
        # which requires that each test has a fixture to accept the parameter.
        # We can add a new fixture like so:
        metafunc.fixturenames.append('tmp_ct')

        # Now we parametrize. This is what happens when we do e.g.,
        # @pytest.mark.parametrize('tmp_ct', range(count))
        # def test_foo(): pass
        metafunc.parametrize('tmp_ct', range(count))
