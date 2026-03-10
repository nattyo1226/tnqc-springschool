using LinearAlgebra

function main()
    # 1. 1-qubit Pauli / Clifford gates
    Id = ComplexF64[
        1 0
        0 1
    ]
    X = ComplexF64[
        0 1
        1 0
    ]
    Y = ComplexF64[
        0 -im
        im 0
    ]
    Z = ComplexF64[
        1 0
        0 -1
    ]
    H = (1 / sqrt(2)) * ComplexF64[
        1 1
        1 -1
    ]
    S = ComplexF64[
        1 0
        0 im
    ]

    # 2-qubit gates (basis order: |00>, |01>, |10>, |11>)
    CNOT = ComplexF64[
        1 0 0 0
        0 1 0 0
        0 0 0 1
        0 0 1 0
    ]
    CZ = Diagonal(ComplexF64[1, 1, 1, -1])

    paulis = [Id, X, Y, Z]
    pauli_labels = ["I", "X", "Y", "Z"]
    signs = [1, -1]
    sign_labels = ["+", "-"]

    # for H gate
    println("H conjugation:")
    h_table = []

    for (i, P) in enumerate(paulis)
        M = H * P * H'

        for (s, sign) in enumerate(signs)
            for (j, Q) in enumerate(paulis)
                if all(M .≈ sign * Q)
                    push!(h_table, (i, s, j))
                    break
                end
            end
        end
    end

    for (i, s, j) in h_table
        println("$(pauli_labels[i]) -> $(sign_labels[s]) $(pauli_labels[j])")
    end
    println()

    # for S gate
    println("S conjugation:")
    s_table = []
    for (i, P) in enumerate(paulis)
        M = S * P * S'

        for (s, sign) in enumerate(signs)
            for (j, Q) in enumerate(paulis)
                if all(M .≈ sign * Q)
                    push!(s_table, (i, s, j))
                    break
                end
            end
        end
    end

    for (i, s, j) in s_table
        println("$(pauli_labels[i]) -> $(sign_labels[s]) $(pauli_labels[j])")
    end
    println()

    # for CNOT gate
    println("CNOT conjugation:")
    cnot_table = []
    for (i0, P0) in enumerate(paulis)
        for (i1, P1) in enumerate(paulis)
            P = kron(P0, P1)
            M = CNOT * P * CNOT'

            for (s, sign) in enumerate(signs)
                for (j0, Q0) in enumerate(paulis)
                    for (j1, Q1) in enumerate(paulis)
                        Q = kron(Q0, Q1)

                        if all(M .≈ sign * Q)
                            push!(cnot_table, (i0, i1, s, j0, j1))
                            break
                        end
                    end
                end
            end
        end
    end

    for (i0, i1, s, j0, j1) in cnot_table
        println(
            "$(pauli_labels[i0])$(pauli_labels[i1]) -> $(sign_labels[s]) $(pauli_labels[j0])$(pauli_labels[j1])",
        )
    end
    println()

    # for CZ gate
    println("CZ conjugation:")
    cz_table = []
    for (i0, P0) in enumerate(paulis)
        for (i1, P1) in enumerate(paulis)
            P = kron(P0, P1)
            M = CZ * P * CZ'

            for (s, sign) in enumerate(signs)
                for (j0, Q0) in enumerate(paulis)
                    for (j1, Q1) in enumerate(paulis)
                        Q = kron(Q0, Q1)

                        if all(M .≈ sign * Q)
                            push!(cz_table, (i0, i1, s, j0, j1))
                            break
                        end
                    end
                end
            end
        end
    end

    for (i0, i1, s, j0, j1) in cz_table
        println(
            "$(pauli_labels[i0])$(pauli_labels[i1]) -> $(sign_labels[s]) $(pauli_labels[j0])$(pauli_labels[j1])",
        )
    end
    println()
end

main()
