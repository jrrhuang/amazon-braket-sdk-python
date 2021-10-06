# general imports
import math
from enum import Enum
from scipy.linalg import cossin, eig

# AWS imports: Import Braket SDK modules
from braket.circuits.composite_operator import CompositeOperator
from braket.circuits.gates import *
from braket.circuits.instruction import Instruction
from braket.circuits.qubit_set import QubitSet
from braket.circuits.synthesis.one_qubit_decomposition import OneQubitDecomposition
from braket.circuits.synthesis.two_qubit_decomposition import (TwoQubitDecomposition,
                                                               two_qubit_decompose)


class QFT_Method(Enum):
    DEFAULT = "default"
    RECURSIVE = "recursive"

    def __str__(self):
        return self.value

    @staticmethod
    def values():
        return list(map(str, QFT_Method))


class Unitary_Method(Enum):
    DEFAULT = "default"
    SHANNON = "shannon"

    def __str__(self):
        return self.value

    @staticmethod
    def values():
        return list(map(str, Unitary_Method))


class GHZ(CompositeOperator):
    """
    Operator for constructing the Greenberger–Horne–Zeilinger state.

    Args:
        qubit_count (int): Number of target qubits.
    """

    def __init__(self, qubit_count: int):
        super().__init__(qubit_count=qubit_count, ascii_symbols=["GHZ"])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the quantum circuit that constructs
        the Greenberger-Horne-Zeilinger (GHZ) state, applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for GHZ

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """
        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = [Instruction(Gate.H(), target=target[0])]
        for i in range(0, len(target) - 1):
            instructions.append(Instruction(Gate.CNot(), target=[target[i], target[i + 1]]))
        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def ghz(targets: QubitSet):
        """
        Registers this function into the circuit class.

        Args:
            targets (QubitSet): Target qubits.

        Returns:
            Instruction: GHZ instruction.

        Examples:
            >>> circ = Circuit().ghz([0, 1, 2])
        """
        return Instruction(CompositeOperator.GHZ(len(targets)), target=targets)


CompositeOperator.register_composite_operator(GHZ)


class QFT(CompositeOperator):
    """
    Operator for quantom fourier transform

    Args:
        qubit_count (int): Number of target qubits.
        method (Union[Enum, str]): Specification of method to use for decomposition,
                         with non-recursive approach by default (method="default"),
                         or recursive approach (method="recursive").
    """

    def __init__(self, qubit_count: int, method=QFT_Method.DEFAULT):

        if str(method) not in QFT_Method.values():
            raise TypeError("method must either be 'default' or 'recursive'.")

        self._method = str(method)
        super().__init__(qubit_count=qubit_count, ascii_symbols=["QFT"])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the Quantum Fourier Transform (QFT)
        algorithm, applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for QFT

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """

        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = []

        # get number of qubits
        num_qubits = len(target)

        if self._method == "recursive":
            # On a single qubit, the QFT is just a Hadamard.
            if len(target) == 1:
                instructions.append(Instruction(Gate.H(), target=target))

            # For more than one qubit, we define mQFT recursively:
            else:

                # First add a Hadamard gate
                instructions.append(Instruction(Gate.H(), target=target[0]))

                # Then apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
                for k, qubit in enumerate(target[1:]):
                    instructions.append(
                        Instruction(
                            Gate.CPhaseShift(2 * math.pi / (2 ** (k + 2))),
                            target=[qubit, target[0]],
                        )
                    )

                # Now apply the above gates recursively to the rest of the qubits
                instructions.append(
                    Instruction(CompositeOperator.mQFT(len(target[1:])), target=target[1:])
                )

        elif self._method == "default":
            for k in range(num_qubits):
                # First add a Hadamard gate
                instructions.append(Instruction(Gate.H(), target=[target[k]]))

                # Then apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
                # Start on the qubit after qubit k, and iterate until the end.  When num_qubits==1, this loop does not run.
                for j in range(1, num_qubits - k):
                    angle = 2 * math.pi / (2 ** (j + 1))
                    instructions.append(
                        Instruction(Gate.CPhaseShift(angle), target=[target[k + j], target[k]])
                    )

        # Then add SWAP gates to reverse the order of the qubits:
        for i in range(math.floor(num_qubits / 2)):
            instructions.append(Instruction(Gate.Swap(), target=[target[i], target[-i - 1]]))

        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def qft(targets: QubitSet, method: str = "default") -> Instruction:
        """Registers this function into the circuit class.

        Args:
            targets (QubitSet): Target qubits.

        Returns:
            Instruction: QFT instruction.

        Examples:
            >>> circ = Circuit().mqft([0, 1, 2])
        """
        return Instruction(CompositeOperator.QFT(len(targets), method), target=targets)


CompositeOperator.register_composite_operator(QFT)


class mQFT(CompositeOperator):
    """
    Operator for "modified" quantom fourier transform. This is the same as quantum fourier transform but
    excluding the SWAP gates.

    Args:
        qubit_count (int): Number of target qubits.
    """

    def __init__(self, qubit_count: int):
        super().__init__(qubit_count=qubit_count, ascii_symbols=["mQFT"])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the Quantum Fourier Transform (QFT)
        algorithm, applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for QFT

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """

        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = []

        if len(target) == 1:
            instructions.append(Instruction(Gate.H(), target=target))

        # For more than one qubit, we define mQFT recursively:
        else:

            # First add a Hadamard gate
            instructions.append(Instruction(Gate.H(), target=target[0]))

            # Then apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
            for k, qubit in enumerate(target[1:]):
                instructions.append(
                    Instruction(
                        Gate.CPhaseShift(2 * math.pi / (2 ** (k + 2))), target=[qubit, target[0]]
                    )
                )

            # Now apply the above gates recursively to the rest of the qubits
            instructions.append(
                Instruction(CompositeOperator.mQFT(len(target[1:])), target=target[1:])
            )

        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def mqft(targets: QubitSet) -> Instruction:
        """Registers this function into the circuit class.

        Args:
            targets (QubitSet): Target qubits.

        Returns:
            Instruction: QFT instruction.

        Examples:
            >>> circ = Circuit().mqft([0, 1, 2])
        """
        return Instruction(CompositeOperator.mQFT(len(targets)), target=targets)


CompositeOperator.register_composite_operator(mQFT)


class iQFT(CompositeOperator):
    """
    Operator for inverse quantom fourier transform

    Args:
        qubit_count (int): Number of target qubits.
    """

    def __init__(self, qubit_count: int):
        super().__init__(qubit_count=qubit_count, ascii_symbols=["iQFT"])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the inverse Quantum Fourier Transform
        (iQFT) algorithm, applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for iQFT

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """

        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = []

        # Set number of qubits
        num_qubits = len(target)

        # First add SWAP gates to reverse the order of the qubits:
        for i in range(math.floor(num_qubits / 2)):
            instructions.append(Instruction(Gate.Swap(), target=[target[i], target[-i - 1]]))

        # Start on the last qubit and work to the first.
        for k in reversed(range(num_qubits)):

            # Apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
            # These angles are the negative of the angle used in the QFT.
            # Start on the last qubit and iterate until the qubit after k.
            # When num_qubits==1, this loop does not run.
            for j in reversed(range(1, num_qubits - k)):
                angle = -2 * math.pi / (2 ** (j + 1))
                instructions.append(
                    Instruction(Gate.CPhaseShift(angle), target=[target[k + j], target[k]])
                )

            # Then add a Hadamard gate
            instructions.append(Instruction(Gate.H(), target=[target[k]]))

        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def iqft(targets: QubitSet) -> Instruction:
        """Registers this function into the circuit class.

        Args:
            targets (QubitSet): Target qubits.

        Returns:
            Instruction: iQFT instruction.

        Examples:
            >>> circ = Circuit().iqft([0, 1, 2])
        """
        return Instruction(CompositeOperator.iQFT(len(targets)), target=targets)


CompositeOperator.register_composite_operator(iQFT)


class QPE(CompositeOperator):
    """
    Operator for Quantum Phase Estimation.

    Args:
        precision_qubit_count (int): The number of qubits in the precision register.
        query_qubit_count (int): The number of qubits in the query register.
        matrix (numpy.ndarray): Unitary matrix whose eigenvalues we wish to estimate.
        control (boolean): Optional boolean flag for controlled unitaries,
                         with C-(U^{2^k}) by default (default is True),
                         or C-U controlled-unitary (2**power) times.

    Raises:
        ValueError: If `matrix` is not a two-dimensional square matrix,
            has a dimension length that is not a positive power of 2 or
            does not match 2 ** (number of query qubits), or is not
            unitary.
    """

    def __init__(
            self, precision_qubit_count: int, query_qubit_count: int, matrix: np.ndarray, control=True
    ):
        Gate.Unitary(matrix)
        self._matrix = np.array(matrix, dtype=complex)

        if len(matrix) != 2 ** query_qubit_count:
            raise ValueError(
                f"dim of matrix {self._matrix} must match the number of query qubits {query_qubit_count}"
            )

        self._condense = control
        self._precision_qubit_count = precision_qubit_count
        self._query_qubit_count = query_qubit_count
        super().__init__(
            qubit_count=precision_qubit_count + query_qubit_count, ascii_symbols=["QPE"]
        )

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def _controlled_unitary(self, unitary) -> np.ndarray:
        # Define projectors onto the computational basis
        p0 = np.array([[1.0, 0.0], [0.0, 0.0]])

        p1 = np.array([[0.0, 0.0], [0.0, 1.0]])

        # Construct numpy matrix
        id_matrix = np.eye(len(unitary))
        controlled_matrix = np.kron(p0, id_matrix) + np.kron(p1, unitary)

        return controlled_matrix

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the Quantum Phase Estimation
        (QPE) algorithm, applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for QPE

        Raises:
            ValueError: If number of qubits in `target` does not equal to total number of precision
                and query qubits.
        """

        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        precision_qubits = target[: self._precision_qubit_count]
        query_qubits = target[
                       self._precision_qubit_count : self._precision_qubit_count + self._query_qubit_count
                       ]

        # Apply Hadamard across precision qubits
        instructions = [Instruction(Gate.H(), target=qubit) for qubit in precision_qubits]

        # Apply controlled unitaries. Start with the last precision_qubit, and end with the first
        for ii, qubit in enumerate(reversed(precision_qubits)):
            # Set power exponent for unitary
            power = ii

            # Alternative 1: Implement C-(U^{2^k})
            if self._condense:
                # Define new unitary with matrix U^{2^k}
                Uexp = np.linalg.matrix_power(self._matrix, 2 ** power)
                CUexp = self._controlled_unitary(Uexp)

                # Apply the controlled unitary C-(U^{2^k})
                instructions.append(
                    Instruction(Gate.Unitary(CUexp), target=[qubit] + list(query_qubits))
                )

            # Alternative 2: One can instead apply controlled-unitary (2**power) times to get C-U^{2^power}
            else:
                for _ in range(2 ** power):
                    CU = self._controlled_unitary(self._matrix)
                    instructions.append(
                        Instruction(Gate.Unitary(CU), target=[qubit] + list(query_qubits))
                    )

        # Apply inverse qft to the precision_qubits
        instructions.append(
            Instruction(CompositeOperator.iQFT(len(precision_qubits)), precision_qubits)
        )

        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def qpe(targets1: QubitSet, targets2: QubitSet, matrix: np.ndarray, control=True):
        """Registers this function into the circuit class.

        Args:
            targets1 (QubitSet): Qubits defining the precision register.
            targets2 (QubitSet): Qubits defining the query register.
            matrix: Unitary matrix whose eigenvalues we wish to estimate
            control: Optional boolean flag for controlled unitaries,
                         with C-(U^{2^k}) by default (default is True),
                         or C-U controlled-unitary (2**power) times.

        Returns:
            Instruction: QPE instruction.

        Raises:
            ValueError: If `matrix` is not a two-dimensional square matrix,
                or has a dimension length that is not compatible with the `targets`,
                or is not unitary.

        Examples:
            >>> circ = Circuit().qpe(QubitSet([0, 1, 2]), QubitSet([4]), np.array([[0, 1], [1, 0]]))
        """
        if 2 ** len(targets2) != matrix.shape[0]:
            raise ValueError(
                "Dimensions of the supplied unitary are incompatible with the query qubits"
            )

        return Instruction(
            CompositeOperator.QPE(len(targets1), len(targets2), matrix, control),
            target=targets1 + targets2,
        )


CompositeOperator.register_composite_operator(QPE)


class MCRy(CompositeOperator):
    """
    Operator for Multi-controlled Ry gate.

    Args:
        qubit_count (int): Number of target qubits.
    """

    def __init__(self, qubit_count: int, angle: float):
        self._angle = angle
        super().__init__(qubit_count=qubit_count, ascii_symbols=["MCRy"])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the Multi-controlled Ry gate,
        applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for MCRy

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """
        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = []

        for i in range(2):
            if len(target) == 2:
                instructions.append(Instruction(Gate.Ry(self._angle), target=target[-1]))
            else:
                instructions.append(Instruction(CompositeOperator.MCRy(), target=[target[1:]]))
            instructions.append(Instruction(Gate.CNot(), target=[target[0], target[-1]]))
        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def mcry(controls: QubitSet, target: QubitInput, angle: float):
        """
        Registers this function into the circuit class.

        Args:
            controls (QubitSet): Control qubits.
            target (Qubit or int): Target qubit index.
            angle (float): Angle in radians.

        Returns:
            Instruction: MCRy instruction.

        Examples:
            >>> circ = Circuit().mcry([1, 2], 3, 0)
        """
        return Instruction(CompositeOperator.MCRy(len(controls) + 1, angle), target=list(controls) + [target])


CompositeOperator.register_composite_operator(MCRy)


class MCRz(CompositeOperator):
    """
    Operator for Multi-controlled Rz gate.

    Args:
        qubit_count (int): Number of target qubits.
    """

    def __init__(self, qubit_count: int, angle: float):
        self._angle = angle
        super().__init__(qubit_count=qubit_count, ascii_symbols=["MCRz"])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the Multi-controlled Rz gate,
        applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for MCRz

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """
        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = []

        for i in range(2):
            if len(target) == 2:
                instructions.append(Instruction(Gate.Rz(self._angle), target=target[-1]))
            else:
                instructions.append(Instruction(CompositeOperator.MCRz(), target=[target[1:]]))
            instructions.append(Instruction(Gate.CNot(), target=[target[0], target[-1]]))
        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def mcrz(controls: QubitSet, target: QubitInput, angle: float):
        """
        Registers this function into the circuit class.

        Args:
            controls (QubitSet): Control qubits.
            target (Qubit or int): Target qubit index.
            angle (float): Angle in radians.

        Returns:
            Instruction: MCRz instruction.

        Examples:
            >>> circ = Circuit().mcrz([1, 2], 3, 0)
        """
        return Instruction(CompositeOperator.MCRz(len(controls) + 1, angle), target=list(controls) + [target])


CompositeOperator.register_composite_operator(MCRz)


class Unitary(CompositeOperator):
    """
    Operator for generic unitary.

    Args:
        qubit_count (int): Number of target qubits.
    """

    def __init__(self, matrix: np.ndarray, display_name: str = "U", method=Unitary_Method.DEFAULT):
        verify_quantum_operator_matrix_dimensions(matrix)
        self._matrix = np.array(matrix, dtype=complex)
        qubit_count = int(np.log2(self._matrix.shape[0]))

        if not is_unitary(self._matrix):
            raise ValueError(f"{self._matrix} is not unitary")

        if str(method) not in Unitary_Method.values():
            raise TypeError("method must either be 'default' or 'shannon'.")

        self._method = str(method)

        super().__init__(qubit_count=qubit_count, ascii_symbols=[display_name])

    def to_ir(self, target: QubitSet):
        return [instr.to_ir() for instr in self.decompose(target)]

    def decompose(self, target: QubitSet) -> Iterable[Instruction]:
        """
        Returns an iterable of instructions corresponding to the Unitary operator,
        applied to the argument qubits.

        Args:
            target (QubitSet): Target qubits

        Returns:
            Iterable[Instruction]: iterable of instructions for Unitary

        Raises:
            ValueError: If number of qubits in `target` does not equal `qubit_count`.
        """
        if len(target) != self.qubit_count:
            raise ValueError(
                f"Operator qubit count {self.qubit_count} must be "
                f"equal to size of target qubit set {target}"
            )

        instructions = []

        if self._method == "shannon":
            dim = self._matrix.shape[0]

            # One-qubit decomposition
            if dim == 2:
                one_qubit_decomp_circ = OneQubitDecomposition(self._matrix).build_circuit(target)
                instructions += one_qubit_decomp_circ.instructions

            # Two-qubit decomposition
            elif dim == 4:
                two_qubit_decomp_circ = TwoQubitDecomposition(self._matrix).build_circuit(target)
                instructions += two_qubit_decomp_circ.instructions

            else:
                u, cs, vdh = cossin(self._matrix, p= dim // 2, q= dim // 2, separate=True)

                d1, v1 = np.linalg.eig(u[0] @ u[1].conj().T)
                d2, v2 = np.linalg.eig(vdh[0] @ vdh[1].conj().T)
                d1 = np.sqrt(d1)
                d2 = np.sqrt(d2)
                v1 = v1.conj().T
                v2 = v2.conj().T

                w1 = d1 @ v1.conj.T() @ u[0]
                w2 = d2 @ v2.conj.T() @ u[1]

                V1 = np.kron(np.eye(2), v1)
                V2 = np.kron(np.eye(2), v2)
                W1 = np.kron(np.eye(2), w1)
                W2 = np.kron(np.eye(2), w2)

                # I made guesses as to what these are
                rz_angle1 = np.log(d1[-1][-1] * 2 / -1j)
                rz_angle2 = np.log(d2[-1][-1] * 2 / -1j)
                ry_angle = np.arccos(cs[-1][-1] * 2)

                instructions += [
                    Instruction(CompositeOperator.Unitary(V1, method="shannon"), target=target[:-1]),
                    Instruction(CompositeOperator.MCRz(len(target), rz_angle1), target=target),
                    Instruction(CompositeOperator.Unitary(W1, method="shannon"), target=target[:-1]),
                    Instruction(CompositeOperator.MCRy(len(target), ry_angle), target=target),
                    Instruction(CompositeOperator.Unitary(V2, method="shannon"), target=target[:-1]),
                    Instruction(CompositeOperator.MCRz(len(target), rz_angle2), target=target),
                    Instruction(CompositeOperator.Unitary(W2, method="shannon"), target=target[:-1])
                ]

        elif self._method == "default":
            instructions.append(Instruction(Gate.Unitary(self._matrix), target=target))

        return instructions

    @staticmethod
    @circuit.subroutine(register=True)
    def unitaryop(targets: QubitSet, matrix: np.ndarray, display_name: str = "U", method: str = "default"):
        """Registers this function into the circuit class.

        Args:
            targets (QubitSet): Target qubits.
            matrix (numpy.ndarray): Unitary matrix which defines the gate. Matrix should be
                compatible with the supplied targets, with `2 ** len(targets) == matrix.shape[0]`.
            display_name (str): Name to be used for an instance of this unitary gate
                for circuit diagrams. Defaults to `U`.

        Returns:
            Instruction: Unitary instruction.

        Raises:
            ValueError: If `matrix` is not a two-dimensional square matrix,
                or has a dimension length that is not compatible with the `targets`,
                or is not unitary,

        Examples:
            >>> circ = Circuit().unitaryop(matrix=np.array([[0, 1],[1, 0]]), targets=[0])
        """
        if 2 ** len(targets) != matrix.shape[0]:
            raise ValueError("Dimensions of the supplied unitary are incompatible with the targets")

        return Instruction(CompositeOperator.Unitary(matrix, display_name, method), target=targets)


CompositeOperator.register_composite_operator(Unitary)
