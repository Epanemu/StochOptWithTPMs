from stochopt.tpms.TreeTPM.histograms import JointHistogram


def test_unify_bins():
    # 1. Identical bins
    bins1 = [{1, 2}, {3, 4}]
    bins2 = [{1, 2}, {3, 4}]
    unified = JointHistogram.unify_bins(bins1, bins2)
    print(f"Test 1 (Identical): {unified}")
    assert len(unified) == 2
    assert {1, 2} in unified
    assert {3, 4} in unified

    # 2. Overlapping but different refinement
    bins1 = [{1, 2, 3}, {4, 5}]
    bins2 = [{1, 2}, {3, 4}, {5}]
    unified = JointHistogram.unify_bins(bins1, bins2)
    print(f"Test 2 (Overlapping): {unified}")
    # Possible groups: {1, 2}, {3}, {4}, {5}
    assert len(unified) == 4
    assert {1, 2} in unified
    assert {3} in unified
    assert {4} in unified
    assert {5} in unified

    # 3. Disjoint
    bins1 = [{1, 2}]
    bins2 = [{3, 4}]
    unified = JointHistogram.unify_bins(bins1, bins2)
    print(f"Test 3 (Disjoint): {unified}")
    assert len(unified) == 2
    assert {1, 2} in unified
    assert {3, 4} in unified

    # 4. One empty
    bins1 = [{1, 2}]
    bins2 = []
    unified = JointHistogram.unify_bins(bins1, bins2)
    print(f"Test 4 (One empty): {unified}")
    assert len(unified) == 1
    assert {1, 2} in unified

    # 5. Missing values covered
    bins1 = [{1}, {2}]
    bins2 = [{2}, {3}]
    unified = JointHistogram.unify_bins(bins1, bins2)
    print(f"Test 5 (Missing values): {unified}")
    # Should have {1}, {2}, {3}
    assert len(unified) == 3
    assert {1} in unified
    assert {2} in unified
    assert {3} in unified

    print("All unify_bins tests passed!")


if __name__ == "__main__":
    test_unify_bins()
