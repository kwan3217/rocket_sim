from matplotlib import pyplot as plt


def pytest_sessionfinish(session, exitstatus):
    """
    This function will be called once after all tests have been executed.
    
    :param session: The pytest session object.
    :param exitstatus: The exit status of the test session.
    """
    # Your code to run once after all tests
    print("All tests have finished running.")
    # Add whatever cleanup, logging, or post-test actions you need here
    plt.show()