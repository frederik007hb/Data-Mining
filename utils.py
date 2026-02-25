def encode_results(results):
    """
    Endcoding of results
    1: HOME WIN, -1: AWAY WIN, 0: DRAW

    Args:
    results: list of dicts - results to encode 
    """
    encoding = np.zeros(len(results))

    for i in range(len(results)):
        if results[i] == "H":
            encoding[i] = 1
        elif results[i] == "A":
            encoding[i] = -1
        else:
            encoding[i] = 0
    return encoding