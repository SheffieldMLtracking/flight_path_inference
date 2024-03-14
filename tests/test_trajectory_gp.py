import flight_path_inference.trajectory_gp


def test_trajectory_gp():
    """
    Test the gaussian process flight inference.

    Based on:
    https://github.com/lionfish0/beelabel/blob/master/labelling/compute5.ipynb
    """
    # Z,mu,scale = tgp.run(obstimes,observations,500,learning_rate=0.1,likenoisescale=0.05,Nind=15)
    # Timestamps (unix epoch seconds)
    obstimes = (1710345107.0, 1710345108.0, 1710345109.0)
    observations = (
        # (originX, originY, originZ, vectorX, vectorY, vectorZ, confidence)
        # Figure this data structure out using the example Jupyter notebook
        # TODO (0.,0.,0., ???)
    )
    flight_path_inference.trajectory_gp.run(obstimes, observations)
