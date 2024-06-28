import numpy as np
from scipy.special import iv
from scipy.constants import mu_0
import matplotlib.pyplot as plt

# import math

# height of cs vs min heat dissipated (from plot6 or 7), also show temp of both coils

# PLOTTING TOGGLES
# NOTE: If all graphs are disabled (False), you will see a blank plot
current_and_voltage_profile = True
heat_dissipation_vs_TFCoil_raii = False
heat_dissipation_vs_CSCoil_thickness = False  # Broken

optimal_heat_dissipation_for_each_coil = False
optimal_temp_change_for_each_coil = False

cs_coil_height_vs_heat_dissipated = False
cs_coil_height_vs_geometry = False
cs_coil_height_vs_temperature = False

show_plots = True  # Whether to display the plots generated (disable this if you only want to save_plots)
save_plots = False # Toggle to export plots to the current working directory (will overwrite if already present, will only save enabled graphs, ignores "show_plots")

# CONSTANTS
copper_resistivity = 1.77e-8  # Copper resistivity in Ohm m
copper_heat_capacity = 8.96e6 * 0.385  # Copper volumetric heat capacity in J/(m^3 K)
neoprene_heat_capacity = 1.23e6 * 1.12  # Neoprene volumetric heat capacity in J/(m^3 K)
radius_outer = 0.4668  # Radius at outer wall of vacuum vessel in m
radius_inner = 0.1428  # Radius at inner wall of vacuum vessel in m

# PLASMA PARAMETERS
plasma_major_radius = 0.22  # Plasma major radius in m
plasma_minor_radius = 0.07  # Plasma minor radius in m
plasma_elongation = 1  # Plasma elongation
magnetic_field_vacuum = 0.1  # Vacuum magnetic field at R0 in T
plasma_current_target = 1e3  # Target plasma current in A
plasma_temperature_target = 50  # Target plasma temperature in eV
breakdown_electric_field = 0.5  # Breakdown electric field at R0 in V/m
plasma_resistance = (
    1.03e-4
    * 2
    * 20
    / plasma_temperature_target ** (3 / 2)
    * plasma_major_radius
    / (plasma_elongation * plasma_minor_radius**2)
)  # Plasma resistance in Ohms
total_flat_top_current = (
    2 * np.pi * plasma_major_radius * magnetic_field_vacuum / mu_0
)  # Total flat-top current through all TF coils in A

# VV PARAMATERS
vv_BoreRadius = 0.1  # Radius from center of machine to inner wall of vacuum vessel in m

# TF COIL PARAMATERS
tfCoil_R0 = plasma_major_radius
# tfCoil_R1 = 0.045  # Radius from origin of machine to inner leg of TF Coil in m
# tfCoil_I0 = 1.516  # Modified bessel function of the first kind, calculated externally
# tfCoil_I1 = 0.847  # Modified bessel function of the second kind, calculated externally
tfCoil_Lminus1 = 1.078  # Modified Struve function, calculated externally
tfCoil_MinimumR1 = (
    0.045  # Minimum radius from center of machine to TF Coil inner leg in m
)

# CS COIL PARAMETERS
csCoil_MaxOuterRadius = (
    vv_BoreRadius - 0.005
)  # Max outer radius of the CS Coil, 5 mm away from the VV
csCoil_MinThickness = 0.01  # Minimum thickness of the CS Coil in m

# PLOTTING PARAMATERS
experiment_time = 0.1
numberOfGraphPoints = 100  # WARNING: Graph resolution, reccommend 100 for fast compute, 10000 for accurate compute
cs_thickness_range = np.linspace(
    0.01, 0.1, numberOfGraphPoints
)  # Range for CS coil thickness in meters
tf_radius_range = np.linspace(
    tfCoil_MinimumR1, csCoil_MaxOuterRadius - csCoil_MinThickness, numberOfGraphPoints
)  # Range for TF radius in meters
cs_coil_heights = np.linspace(0.2, 1.0, 5)  # Range for CS coil heights in meters


class TF:
    def __init__(
        self,
        inner_radius_dee,
        num_windings,
        num_coils=8,
        fill_factor=0.9,
        copper_volume_fraction=0.5,
    ):
        """
        Parameters
        ----------
        inner_radius_dee:
            Princeton Dee inner radius in m.

        num_windings:
            Number of windings per coil.

        num_coils:
            Number of coils.

        fill_factor:
            Circle packing fill factor.

        copper_volume_fraction:
            Volume percentage of copper per wire.
        """
        if inner_radius_dee < plasma_major_radius:
            self.inner_radius_dee = inner_radius_dee
        else:
            raise ValueError(
                f"Inner radius ({inner_radius_dee}) must be less than plasma major radius ({plasma_major_radius})."
            )

        self.num_windings = num_windings
        self.num_coils = num_coils
        self.fill_factor = fill_factor
        self.copper_volume_fraction = copper_volume_fraction

        self.outer_radius_dee = (
            plasma_major_radius**2 / self.inner_radius_dee
        )  # Outer TF radius in m
        self.dee_shape_factor = (
            1 / 2 * np.log(self.outer_radius_dee / self.inner_radius_dee)
        )  # Dee shape factor

        self.cross_sectional_area = (
            np.pi
            * self.inner_radius_dee**2
            * fill_factor
            / (self.num_windings * self.num_coils)
        )  # Winding cross sectional area in m^2
        self.total_coil_length = (
            2
            * np.pi
            * self.dee_shape_factor
            * plasma_major_radius
            * self.num_windings
            * (iv(0, self.dee_shape_factor) + iv(1, self.dee_shape_factor))
        )  # Total coil length in m

        self.resistance = (
            copper_resistivity
            * self.total_coil_length
            / (copper_volume_fraction * self.cross_sectional_area)
        )  # Total coil resistance on Ohms
        self.inductance = (
            2
            * np.pi
            * mu_0
            * self.dee_shape_factor**2
            * plasma_major_radius
            * self.num_windings**2
            * (
                iv(0, self.dee_shape_factor)
                + 2 * iv(1, self.dee_shape_factor)
                + iv(2, self.dee_shape_factor)
            )
        )  # Total coil inductance in H
        self.effective_heat_capacity = (
            self.total_coil_length
            * self.cross_sectional_area
            * (
                copper_volume_fraction * copper_heat_capacity
                + (1 - copper_volume_fraction) * neoprene_heat_capacity
            )
        )  # Effective volumetric heat capacity in J/(m^3 K)

    def reset_geometry(
        self,
        inner_radius_dee,
        num_windings,
        num_coils=8,
        fill_factor=0.9,
        copper_volume_fraction=0.5,
    ):
        self.__init__(
            inner_radius_dee,
            num_windings,
            num_coils,
            fill_factor,
            copper_volume_fraction,
        )

    def set_current_profile(
        self, ramp_up_time=10e-3, experiment_time=experiment_time, ramp_down_time=10e-3
    ):
        """
        Parameters
        ----------
        ramp_up_time:
            Ramp up time in s (Default = 10e-3)

        experiment_time:
            Experiment time in s (Default = 0.1)

        ramp_down_time:
            Ramp down time in s (Default = 10e-3)
        """
        max_current = total_flat_top_current / (
            self.num_windings * self.num_coils
        )  # Maximum TF current in A

        time_points = (
            0,
            ramp_up_time,
            ramp_up_time + experiment_time,
            ramp_up_time + experiment_time + ramp_down_time,
        )
        current_points = (0, max_current, max_current, 0)
        time, current, current_rate, current_integral = create_current_profile(
            time_points, current_points
        )

        self.time = (
            time - ramp_up_time
        )  # Time, shifted so that t=0 corresponds to start of experiment
        self.current = current  # TF current
        self.voltage = (
            self.current * self.resistance + self.inductance * current_rate
        )  # TF voltage
        self.heat_dissipated = (
            current_integral * self.resistance * self.num_coils
        )  # Heat dissipated
        self.temperature_change = (
            self.heat_dissipated / self.effective_heat_capacity
        )  # Change in temperature


class CS:
    def __init__(
        self,
        radius_solenoid,
        height_solenoid,
        thickness_solenoid,
        num_windings,
        fill_factor=0.9,
        copper_volume_fraction=0.5,
    ):
        """
        Parameters
        ----------
        radius_solenoid:
            Central solenoid radius in m.

        height_solenoid:
            Central solenoid height in m.

        thickness_solenoid:
            Central solenoid thickness in m.

        num_windings:
            Number of windings.

        fill_factor:
            Circle packing fill factor.

        copper_volume_fraction:
            Volume percentage of copper per wire.
        """
        if radius_solenoid + thickness_solenoid / 2 < plasma_major_radius:
            self.radius_solenoid = radius_solenoid
        else:
            raise ValueError(
                f"(Radius ({radius_solenoid})+ thickness ({thickness_solenoid})) must be less than plasma major radius ({plasma_major_radius})."
            )

        self.height_solenoid = height_solenoid
        self.thickness_solenoid = thickness_solenoid
        self.num_windings = num_windings
        self.fill_factor = fill_factor
        self.copper_volume_fraction = copper_volume_fraction

        self.cross_sectional_area = (
            self.height_solenoid * self.thickness_solenoid * fill_factor / num_windings
        )  # Winding cross sectional area in m^2
        self.total_coil_length = (
            2 * np.pi * self.radius_solenoid * num_windings
        )  # Total coil length in m

        self.resistance = (
            copper_resistivity
            * self.total_coil_length
            / (copper_volume_fraction * self.cross_sectional_area)
        )  # Total coil resistance on Ohms
        self.inductance = (
            mu_0
            * num_windings**2
            * np.pi
            * self.radius_solenoid**2
            / self.height_solenoid
        )  # Total coil inductance in H
        self.effective_heat_capacity = (
            self.total_coil_length
            * self.cross_sectional_area
            * (
                copper_volume_fraction * copper_heat_capacity
                + (1 - copper_volume_fraction) * neoprene_heat_capacity
            )
        )  # Effective volumetric heat capacity in J/(m^3 K)

        self.breakdown_current_rate = (
            2
            * plasma_major_radius
            * self.height_solenoid
            / (mu_0 * self.radius_solenoid**2)
            * breakdown_electric_field
            / num_windings
        )  # Breakdown discharge rate in A/s
        self.mutual_inductance = (
            mu_0
            * np.pi
            * num_windings
            * self.radius_solenoid**2
            / (2 * plasma_major_radius)
        )  # Mutual inductance in H
        self.flat_top_current_rate = (
            plasma_resistance * plasma_current_target / self.mutual_inductance
        )  # Flat top discharge rate in A/s

    def reset_geometry(
        self,
        radius_solenoid,
        height_solenoid,
        thickness_solenoid,
        num_windings,
        fill_factor=0.9,
        copper_volume_fraction=0.5,
    ):
        self.__init__(
            radius_solenoid,
            height_solenoid,
            thickness_solenoid,
            num_windings,
            fill_factor,
            copper_volume_fraction,
        )

    def set_current_profile(
        self,
        ramp_up_time=10e-3,
        prepulse_time=10e-3,
        breakdown_time=10e-3,
        experiment_time=experiment_time,
        ramp_down_time=10e-3,
    ):
        """
        Parameters
        ----------
        ramp_up_time:
            Ramp up time in s

        prepulse_time:
            Prepulse time in s

        breakdown_time:
            Break down time in s

        experiment_time:
            Experiment time in s

        ramp_down_time:
            Ramp down time in s
        """
        delta_current_breakdown = (
            self.breakdown_current_rate * breakdown_time
        )  # Change in current during breakdown

        delta_current_flattop = (
            self.flat_top_current_rate * experiment_time
        )  # Change in current during flattop
        max_current = (
            delta_current_breakdown + delta_current_flattop
        ) / 2  # Maximum current

        time_points = (
            0,
            ramp_up_time,
            ramp_up_time + prepulse_time,
            ramp_up_time + prepulse_time + breakdown_time,
            ramp_up_time + prepulse_time + breakdown_time + experiment_time,
            ramp_up_time
            + prepulse_time
            + breakdown_time
            + experiment_time
            + ramp_down_time,
        )
        current_points = (
            0,
            max_current,
            max_current,
            max_current - delta_current_breakdown,
            -max_current,
            0,
        )
        time, current, current_rate, current_integral = create_current_profile(
            time_points, current_points
        )

        self.time = (
            time - ramp_up_time - prepulse_time
        )  # Time, shifted so that t=0 corresponds to start of experiment
        self.current = current  # TF current
        self.voltage = (
            self.current * self.resistance + self.inductance * current_rate
        )  # TF voltage
        self.heat_dissipated = current_integral * self.resistance  # Heat dissipated
        self.temperature_change = (
            self.heat_dissipated / self.effective_heat_capacity
        )  # Change in temperature


# -------------------PLOTTING---------------------
def create_current_profile(time_points, current_points):
    """
    Create a current profile from points.
    """
    dt = np.min(np.diff(time_points)) / 1000
    time = np.ravel([[t_point - dt, t_point + dt] for t_point in time_points[1:-1]])
    time = np.insert(time, [0, len(time)], [time_points[0], time_points[-1]])

    current = np.ravel([[i_point, i_point] for i_point in current_points[1:-1]])
    current = np.insert(
        current, [0, len(current)], [current_points[0], current_points[-1]]
    )

    current_rate = np.diff(current_points) / np.diff(time_points)
    current_rate = np.repeat(current_rate, 2)

    current_integral = (
        1
        / 3
        * np.diff(time)
        * (current[1:] ** 2 + current[:-1] ** 2 + current[1:] * current[:-1])
    )  # Integral of i^2
    current_integral = np.sum(current_integral)

    return time, current, current_rate, current_integral


# CLASS IMPLEMENTATIONS OF CS AND TF COIL
# Class initialisation for TF coil and CS Coil
toroidalFieldCoil = TF(0.1, 1, 1)  # Default: TF(0.1, 10, 10)
toroidalFieldCoil.set_current_profile()

centralSolenoid = CS(0.0713, 0.5, 0.0187, 200)  # Default: CS(0.12, 0.5, 20e-3, 1)
centralSolenoid.set_current_profile()


# Define the heat_dissipation_function
def heat_dissipation_function(
    tf_coil_instance, cs_coil_instance, new_tf_radius, new_cs_radius
):
    # Create new instances with updated radii but other parameters preserved
    new_toroidalFieldCoil = TF(
        new_tf_radius,
        tf_coil_instance.num_windings,
        tf_coil_instance.num_coils,
    )
    new_centralSolenoid = CS(
        new_cs_radius,
        cs_coil_instance.height_solenoid,
        cs_coil_instance.thickness_solenoid,
        cs_coil_instance.num_windings,
    )

    # Re-calculate current profiles and hence heat dissipation
    new_toroidalFieldCoil.set_current_profile()
    new_centralSolenoid.set_current_profile()

    # Calculate heat dissipation
    total_heat_dissipation = (
        new_toroidalFieldCoil.heat_dissipated + new_centralSolenoid.heat_dissipated
    )
    tf_heat_dissipation = new_toroidalFieldCoil.heat_dissipated
    cs_heat_dissipation = new_centralSolenoid.heat_dissipated

    # Get change in temperature for each coil
    new_tfTempChange = new_toroidalFieldCoil.temperature_change
    new_csTempChange = new_centralSolenoid.temperature_change

    return total_heat_dissipation, tf_heat_dissipation, cs_heat_dissipation


def get_csCoilThickness_given_tfCoil_R1(tfCoil_R1):
    csCoil_Thickness = csCoil_MaxOuterRadius - 0.005 - tfCoil_R1
    return csCoil_Thickness


def get_tfCoil_R1_given_csCoilThickness(csCoilThickness):
    tfCoil_R1 = csCoil_MaxOuterRadius - 0.005 - csCoilThickness
    return tfCoil_R1


# Function for optimising heat dissipation and temperature change for different CS Heights and TF inner leg radii
def get_heat_dissipated_and_temperature_change_from_tfCoil_R1(
    cs_coil_instance, tf_coil_instance, tfCoil_R1, csCoil_Height
):
    # Create new instances with updated radii but other parameters preserved
    new_toroidalFieldCoil = TF(
        tfCoil_R1,  # Radius from center of machine to inner leg of TF Coil
        tf_coil_instance.num_windings,
        tf_coil_instance.num_coils,
    )
    new_centralSolenoid = CS(
        csCoil_MaxOuterRadius,  # Radius of the CS Coil
        csCoil_Height,  # Height of the CS Coil
        get_csCoilThickness_given_tfCoil_R1(tfCoil_R1),  # Thickness of the CS Coil
        cs_coil_instance.num_windings,
    )

    # Re-calculate current profiles and hence heat dissipation
    new_toroidalFieldCoil.set_current_profile()
    new_centralSolenoid.set_current_profile()

    # Calculate total heat dissipation
    total_heat_dissipation = (
        new_toroidalFieldCoil.heat_dissipated + new_centralSolenoid.heat_dissipated
    )

    # Get change in temperature for each coil
    new_tfTempChange = new_toroidalFieldCoil.temperature_change
    new_csTempChange = new_centralSolenoid.temperature_change

    return total_heat_dissipation, new_tfTempChange, new_csTempChange


# Function for optimising heat dissipation and temperature change for different CS Heights and CS Coil Thicknessess
def get_heat_dissipated_and_temperature_change_from_csCoil_Thickness(
    cs_coil_instance, tf_coil_instance, csCoil_Thickness, csCoil_Height
):
    # Create new instances with updated radii but other parameters preserved
    new_toroidalFieldCoil = TF(
        get_tfCoil_R1_given_csCoilThickness(
            csCoil_Thickness
        ),  # Radius from center of machine to inner leg of TF Coil
        tf_coil_instance.num_windings,
        tf_coil_instance.num_coils,
    )
    new_centralSolenoid = CS(
        csCoil_MaxOuterRadius,  # Radius of the CS Coil
        csCoil_Height,  # Height of the CS Coil
        csCoil_Thickness,  # Thickness of the CS Coil
        cs_coil_instance.num_windings,
    )

    # Re-calculate current profiles and hence heat dissipation
    new_toroidalFieldCoil.set_current_profile()
    new_centralSolenoid.set_current_profile()

    # Calculate total heat dissipation
    total_heat_dissipation = (
        new_toroidalFieldCoil.heat_dissipated + new_centralSolenoid.heat_dissipated
    )

    # Get change in temperature for each coil
    new_tfTempChange = new_toroidalFieldCoil.temperature_change
    new_csTempChange = new_centralSolenoid.temperature_change

    return total_heat_dissipation, new_tfTempChange, new_csTempChange


if current_and_voltage_profile:
    # Plot 1
    # Create a figure for the plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Plot current vs. time for centralSolenoid in blue
    ax1.plot(
        centralSolenoid.time, centralSolenoid.current, color="blue", label="Current (A)"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Add grid to the plot
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Create a second y-axis for voltage
    ax2 = ax1.twinx()
    ax2.plot(
        centralSolenoid.time, centralSolenoid.voltage, color="red", label="Voltage (V)"
    )
    ax2.set_ylabel("Voltage (V)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add titles and labels
    ax1.set_title("Central Solenoid Current and Voltage vs. Time")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("CurrentAndVoltageProfile.png")

    # Create a separate figure for the tables
    fig2, (ax_table_cs, ax_table_time_points) = plt.subplots(2, 1, figsize=(10, 8))
    ax_table_cs.axis("tight")
    ax_table_cs.axis("off")
    ax_table_time_points.axis("tight")
    ax_table_time_points.axis("off")

    # Format values to 3 significant figures
    def format_value(value):
        return f"{value:.3g}"

    # Table data for CS class parameters
    cs_table_data = [
        ["Parameter", "Value"],
        ["Radius (m)", format_value(centralSolenoid.radius_solenoid)],
        ["Height (m)", format_value(centralSolenoid.height_solenoid)],
        ["Thickness (m)", format_value(centralSolenoid.thickness_solenoid)],
        ["Number of Windings", centralSolenoid.num_windings],
        [
            "Cross-sectional Area (m^2)",
            format_value(centralSolenoid.cross_sectional_area),
        ],
        ["Total Coil Length (m)", format_value(centralSolenoid.total_coil_length)],
        ["Resistance (Ohms)", format_value(centralSolenoid.resistance)],
        ["Inductance (H)", format_value(centralSolenoid.inductance)],
        [
            "Effective Heat Capacity (J/(m^3 K))",
            format_value(centralSolenoid.effective_heat_capacity),
        ],
        [
            "Breakdown Current Rate (A/s)",
            format_value(centralSolenoid.breakdown_current_rate),
        ],
        ["Mutual Inductance (H)", format_value(centralSolenoid.mutual_inductance)],
        [
            "Flat Top Current Rate (A/s)",
            format_value(centralSolenoid.flat_top_current_rate),
        ],
    ]

    # Add CS class parameters table to the plot
    table_cs = ax_table_cs.table(
        cellText=cs_table_data,
        colLabels=None,
        cellLoc="center",
        loc="center",
    )

    # Format the CS class parameters table
    table_cs.auto_set_font_size(False)
    table_cs.set_fontsize(10)
    table_cs.scale(1.2, 1.2)

    # Table data for time points from set_current_profile
    time_points_table_data = [
        ["Time Points", "Value (s)"],
        ["Start", format_value(centralSolenoid.time[0])],
        ["Ramp Up", format_value(centralSolenoid.time[1])],
        ["Prepulse", format_value(centralSolenoid.time[2])],
        ["Breakdown", format_value(centralSolenoid.time[3])],
        ["Experiment", format_value(centralSolenoid.time[4])],
        ["Ramp Down", format_value(centralSolenoid.time[5])],
        ["End", format_value(centralSolenoid.time[-1])],
    ]

    # Add time points table to the plot
    table_time_points = ax_table_time_points.table(
        cellText=time_points_table_data,
        colLabels=None,
        cellLoc="center",
        loc="center",
    )

    # Format the time points table
    table_time_points.auto_set_font_size(False)
    table_time_points.set_fontsize(10)
    table_time_points.scale(1.2, 1.2)

    plt.tight_layout()

    if save_plots:
        plt.savefig("CentralSolenoidParametersAndTimePointsTable.png")

if heat_dissipation_vs_TFCoil_raii:
    # Plot 2
    # Create a new figure for the heat dissipation graph
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Calculate and plot heat dissipation for each CS coil height
    for cs_height in cs_coil_heights:
        heat_dissipations = []

        for tf_radius in tf_radius_range:
            total_heat_dissipation, _, _ = (
                get_heat_dissipated_and_temperature_change_from_tfCoil_R1(
                    centralSolenoid, toroidalFieldCoil, tf_radius, cs_height
                )
            )
            heat_dissipations.append(total_heat_dissipation)

        ax2.plot(
            tf_radius_range, heat_dissipations, label=f"CS Height = {cs_height:.2f} m"
        )

    # Configure plot
    ax2.set_xlabel("TF Radius (m)")
    ax2.set_ylabel("Total Heat Dissipated (J)")
    ax2.set_title(
        "Heat Dissipated as a Function of TF Radius for Different CS Coil Heights"
    )
    ax2.legend()
    ax2.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("HeatDissipationVsTFCoilRadii.png")


if heat_dissipation_vs_CSCoil_thickness:
    # Plot 3
    # Create a new figure for the heat dissipation graph
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Calculate and plot heat dissipation for each CS coil height
    for cs_height in cs_coil_heights:
        heat_dissipations = []

        for cs_thickness in cs_thickness_range:
            # Update CS coil instance with the current thickness and height
            centralSolenoid.reset_geometry(
                centralSolenoid.radius_solenoid,
                cs_height,
                cs_thickness,
                centralSolenoid.num_windings,
            )

            total_heat_dissipation, _, _ = (
                get_heat_dissipated_and_temperature_change_from_csCoil_Thickness(
                    centralSolenoid, toroidalFieldCoil, cs_thickness, cs_height
                )
            )
            heat_dissipations.append(total_heat_dissipation)

        ax3.plot(
            cs_thickness_range,
            heat_dissipations,
            label=f"CS Height = {cs_height:.2f} m",
        )

    # Configure plot
    ax3.set_xlabel("CS Coil Thickness (m)")
    ax3.set_ylabel("Total Heat Dissipated (J)")
    ax3.set_title(
        "Heat Dissipated as a Function of CS Coil Thickness for Different CS Coil Heights"
    )
    ax3.legend()
    ax3.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("HeatDissipationVsCSCoilThickness.png")

# # Plot 4: Temperature change as a function of TF inner leg radius for different CS coil heights
# fig4, ax4 = plt.subplots(figsize=(10, 6))
#
# # Calculate and plot temperature change for each CS coil height
# for cs_height in cs_coil_heights:
#     tf_temp_changes = []
#     cs_temp_changes = []
#
#     for tf_radius in tf_radius_range:
#         total_heat_dissipation, tf_temp_change, cs_temp_change = get_heat_dissipated_and_temperature_change_from_tfCoil_R1(
#             centralSolenoid, toroidalFieldCoil, tf_radius, cs_height
#         )
#         tf_temp_changes.append(tf_temp_change)
#         cs_temp_changes.append(cs_temp_change)
#
#     ax4.plot(tf_radius_range, tf_temp_changes, label=f'TF Temp Change, CS Height = {cs_height:.2f} m')
#     ax4.plot(tf_radius_range, cs_temp_changes, linestyle='--', label=f'CS Temp Change, CS Height = {cs_height:.2f} m')
#
# # Configure plot
# ax4.set_xlabel('TF Inner Leg Radius (m)')
# ax4.set_ylabel('Temperature Change (K)')
# ax4.set_title('Temperature Change as a Function of TF Inner Leg Radius for Different CS Coil Heights')
# ax4.legend()
# ax4.grid(True)
#
# # Plot 5: Temperature change as a function of CS coil thickness for different CS coil heights
# fig5, ax5 = plt.subplots(figsize=(10, 6))
#
# # Calculate and plot temperature change for each CS coil height
# for cs_height in cs_coil_heights:
#     tf_temp_changes = []
#     cs_temp_changes = []
#
#     for cs_thickness in cs_thickness_range:
#         total_heat_dissipation, tf_temp_change, cs_temp_change = get_heat_dissipated_and_temperature_change_from_csCoil_Thickness(
#             centralSolenoid, toroidalFieldCoil, cs_thickness, cs_height
#         )
#         tf_temp_changes.append(tf_temp_change)
#         cs_temp_changes.append(cs_temp_change)
#
#     ax5.plot(cs_thickness_range, tf_temp_changes, label=f'TF Temp Change, CS Height = {cs_height:.2f} m')
#     ax5.plot(cs_thickness_range, cs_temp_changes, linestyle='--', label=f'CS Temp Change, CS Height = {cs_height:.2f} m')
#
# # Configure plot
# ax5.set_xlabel('CS Coil Thickness (m)')
# ax5.set_ylabel('Temperature Change (K)')
# ax5.set_title('Temperature Change as a Function of CS Coil Thickness for Different CS Coil Heights')
# ax5.legend()
# ax5.grid(True)


if optimal_heat_dissipation_for_each_coil:
    # Plot 6
    # Create a new figure for the heat dissipation graph
    fig6, ax6 = plt.subplots(figsize=(10, 6))

    # Calculate and plot heat dissipation for each TF coil radius
    tf_heat_dissipations = []
    cs_heat_dissipations = []
    total_heat_dissipations = []
    tf_and_cs_labels = []

    for tf_radius in tf_radius_range:
        cs_thickness = get_csCoilThickness_given_tfCoil_R1(tf_radius)
        cs_radius = csCoil_MaxOuterRadius - cs_thickness
        tf_and_cs_labels.append(f"{tf_radius:.2f} ({cs_thickness:.2f})")

        # Update TF and CS coil instances with the current radii
        toroidalFieldCoil.reset_geometry(tf_radius, toroidalFieldCoil.num_windings)
        centralSolenoid.reset_geometry(
            cs_radius,
            centralSolenoid.height_solenoid,
            cs_thickness,
            centralSolenoid.num_windings,
        )

        # Calculate heat dissipation for TF and CS coils
        total_heat_dissipation, tf_heat_dissipation, cs_heat_dissipation = (
            heat_dissipation_function(
                toroidalFieldCoil, centralSolenoid, tf_radius, cs_radius
            )
        )

        tf_heat_dissipations.append(tf_heat_dissipation)
        cs_heat_dissipations.append(cs_heat_dissipation)
        total_heat_dissipations.append(total_heat_dissipation)

    # Convert the TF radii to string format for x-axis labeling
    x_labels = np.arange(len(tf_and_cs_labels))

    # Select 20 evenly spaced labels
    num_labels = 20
    step = len(tf_and_cs_labels) // num_labels
    selected_indices = np.arange(0, len(tf_and_cs_labels), step)

    # Find index where TF and CS heat dissipation are minimised
    equal_idx = np.argmin(
        np.abs(np.array(tf_heat_dissipations) + np.array(cs_heat_dissipations))
    )

    # Plotting
    ax6.plot(x_labels, tf_heat_dissipations, label="TF Coil Heat Dissipation")
    ax6.plot(x_labels, cs_heat_dissipations, label="CS Coil Heat Dissipation")
    ax6.plot(x_labels, total_heat_dissipations, label="Total Heat Dissipation")

    # Vertical line at the point of minimum heat dissipation
    ax6.axvline(
        x=equal_idx, color="red", linestyle="--", label="Minimum Coil Heat Dissipation"
    )
    ax6.text(
        equal_idx + 5,
        0.95 * max(total_heat_dissipations),
        f"TF Radius = {tf_radius_range[equal_idx]:.4f} m\nCS Thickness = {get_csCoilThickness_given_tfCoil_R1(tf_radius_range[equal_idx]):.4f} m",
        color="red",
        rotation=0,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3"),
    )

    # Configure plot
    ax6.set_xlabel("TF Radius (m) with Corresponding CS Thickness (m)")
    ax6.set_ylabel("Heat Dissipated (J)")
    ax6.set_title("Heat Dissipated as a Function of TF Radius and CS Coil Thickness")
    ax6.set_xticks(selected_indices)
    ax6.set_xticklabels(
        np.array(tf_and_cs_labels)[selected_indices], rotation=45, ha="right"
    )
    ax6.legend()
    ax6.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("OptimalHeatDissipationForEachCoil.png")


if optimal_temp_change_for_each_coil:
    # Plot 7
    # Create a new figure for the temperature change graph
    fig7, ax7 = plt.subplots(figsize=(10, 6))

    # Calculate and plot temperature change for each TF coil radius
    tf_temp_changes = []
    cs_temp_changes = []
    total_temp_changes = []
    tf_and_cs_labels = []

    for tf_radius in tf_radius_range:
        cs_thickness = get_csCoilThickness_given_tfCoil_R1(tf_radius)
        cs_radius = csCoil_MaxOuterRadius - cs_thickness

        # Update TF and CS coil instances with the current radii
        toroidalFieldCoil.reset_geometry(tf_radius, toroidalFieldCoil.num_windings)
        centralSolenoid.reset_geometry(
            cs_radius,
            centralSolenoid.height_solenoid,
            cs_thickness,
            centralSolenoid.num_windings,
        )

        _, tf_temp_change, cs_temp_change = (
            get_heat_dissipated_and_temperature_change_from_csCoil_Thickness(
                centralSolenoid,
                toroidalFieldCoil,
                cs_thickness,
                centralSolenoid.height_solenoid,
            )
        )

        tf_temp_changes.append(tf_temp_change)
        cs_temp_changes.append(cs_temp_change)
        total_temp_changes.append(tf_temp_change + cs_temp_change)
        tf_and_cs_labels.append(f"{tf_radius:.2f} ({cs_thickness:.2f})")

    # Convert the TF radii to string format for x-axis labeling
    x_labels = np.arange(len(tf_radius_range))

    # Select 20 evenly spaced labels
    num_labels = 20
    step = len(tf_radius_range) // num_labels
    selected_indices = np.arange(0, len(tf_radius_range), step)

    # Find index where TF and CS coil temperature changes are minimal
    equal_idx = np.argmin(np.abs(np.array(tf_temp_changes) + np.array(cs_temp_changes)))

    # Plotting
    ax7.plot(x_labels, tf_temp_changes, label="TF Coil Temperature Change")
    ax7.plot(x_labels, cs_temp_changes, label="CS Coil Temperature Change")
    ax7.plot(
        x_labels, total_temp_changes, label="Total Temperature Change", linestyle="--"
    )

    # Plot red vertical line at the point where TF and CS temperature changes are minimised
    ax7.axvline(
        equal_idx, color="red", linestyle="--", label="Minimum Coil Temp Change"
    )

    # Annotation for equal temperature change point
    ax7.text(
        equal_idx + 5,
        0.95 * max(total_temp_changes),
        f"TF Radius = {tf_radius_range[equal_idx]:.4f}m\nCS Thickness = {get_csCoilThickness_given_tfCoil_R1(tf_radius_range[equal_idx]):.4f}m",
        color="red",
        va="top",  # Vertical alignment centered
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3"),
    )

    # Configure plot
    ax7.set_xlabel("TF Radius (m) with Corresponding CS Thickness (m)")
    ax7.set_ylabel("Temperature Change (K)")
    ax7.set_title("Temperature Change as a Function of TF and CS Coil Radii")
    ax7.set_xticks(selected_indices)
    ax7.set_xticklabels(
        np.array(tf_and_cs_labels)[selected_indices], rotation=45, ha="right"
    )
    ax7.legend()
    ax7.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("OptimalTempChangeForEachCoil.png")


if cs_coil_height_vs_heat_dissipated:
    # Plot 9
    # Create a new figure for Plot 9
    fig9, ax9 = plt.subplots(figsize=(10, 6))

    # To store the minimum heat dissipation for each CS coil height
    min_heat_dissipations = []

    # Loop over the range of CS coil heights
    for cs_height in cs_coil_heights:
        tf_heat_dissipations = []
        cs_heat_dissipations = []
        total_heat_dissipations = []

        for tf_radius in tf_radius_range:
            cs_thickness = get_csCoilThickness_given_tfCoil_R1(tf_radius)
            cs_radius = csCoil_MaxOuterRadius - cs_thickness

            # Update TF and CS coil instances with the current radii
            toroidalFieldCoil.reset_geometry(tf_radius, toroidalFieldCoil.num_windings)
            centralSolenoid.reset_geometry(
                cs_radius,
                cs_height,
                cs_thickness,
                centralSolenoid.num_windings,
            )

            # Calculate heat dissipation for TF and CS coils
            total_heat_dissipation, tf_heat_dissipation, cs_heat_dissipation = (
                heat_dissipation_function(
                    toroidalFieldCoil, centralSolenoid, tf_radius, cs_radius
                )
            )

            tf_heat_dissipations.append(tf_heat_dissipation)
            cs_heat_dissipations.append(cs_heat_dissipation)
            total_heat_dissipations.append(total_heat_dissipation)

        # Find the minimum total heat dissipation for the current CS coil height
        min_total_heat_dissipation = min(total_heat_dissipations)
        min_heat_dissipations.append(min_total_heat_dissipation)

    # Plot the minimum heat dissipation for each CS coil height
    ax9.plot(
        cs_coil_heights,
        min_heat_dissipations,
        marker="o",
        label="Minimum Heat Dissipation",
    )

    # Configure plot
    ax9.set_xlabel("CS Coil Height (m)")
    ax9.set_ylabel("Minimum Heat Dissipated (J)")
    ax9.set_title("Minimum Heat Dissipated as a Function of CS Coil Height")
    ax9.legend()
    ax9.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("MinimumHeatDissipation_vs_CSCoilHeight.png")


if cs_coil_height_vs_geometry:
    # Plot 10
    # Create a new figure for Plot 10
    fig10, ax10 = plt.subplots(figsize=(10, 6))

    # Lists to store the calculated values
    inner_radius_dee = []
    cs_coil_radii_at_min = []
    cs_coil_thicknesses_at_min = []
    outer_radius_dee = []

    # Loop over the range of CS coil heights
    for cs_height in cs_coil_heights:
        total_heat_dissipations = []
        tf_radii = []
        cs_radii = []
        cs_thicknesses = []
        tf_outer_leg_radii = []

        for tf_radius in tf_radius_range:
            cs_thickness = get_csCoilThickness_given_tfCoil_R1(tf_radius)
            cs_radius = csCoil_MaxOuterRadius - cs_thickness

            # Update TF and CS coil instances with the current radii
            toroidalFieldCoil.reset_geometry(tf_radius, toroidalFieldCoil.num_windings)
            centralSolenoid.reset_geometry(
                cs_radius,
                cs_height,
                cs_thickness,
                centralSolenoid.num_windings,
            )

            # Calculate heat dissipation for TF and CS coils
            total_heat_dissipation, tf_heat_dissipation, cs_heat_dissipation = (
                heat_dissipation_function(
                    toroidalFieldCoil, centralSolenoid, tf_radius, cs_radius
                )
            )

            total_heat_dissipations.append(total_heat_dissipation)
            tf_radii.append(tf_radius)
            cs_radii.append(cs_radius)
            cs_thicknesses.append(cs_thickness)
            tf_outer_leg_radii.append(toroidalFieldCoil.outer_radius_dee)

        # Find the index of the minimum total heat dissipation for the current CS coil height
        min_index = np.argmin(total_heat_dissipations)
        inner_radius_dee.append(tf_radii[min_index])
        cs_coil_radii_at_min.append(cs_radii[min_index])
        cs_coil_thicknesses_at_min.append(cs_thicknesses[min_index])
        outer_radius_dee.append(tf_outer_leg_radii[min_index])

    # Plot the values against the CS coil heights
    ax10.plot(
        cs_coil_heights, inner_radius_dee, label="TF Inner Radius Dee", marker="o"
    )
    ax10.plot(cs_coil_heights, cs_coil_radii_at_min, label="CS Coil Radius", marker="x")
    ax10.plot(
        cs_coil_heights,
        cs_coil_thicknesses_at_min,
        label="CS Coil Thickness",
        marker="s",
    )
    ax10.plot(
        cs_coil_heights, outer_radius_dee, label="TF Outer Radius Dee", marker="d"
    )

    # Configure plot
    ax10.set_xlabel("CS Coil Height (m)")
    ax10.set_ylabel("Dimensions (m)")
    ax10.set_title("Dimensions as a Function of CS Coil Height")
    ax10.legend()
    ax10.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("Dimensions_vs_CSCoilHeight.png")


if cs_coil_height_vs_temperature:
    # Plot 11
    # Create a new figure for Plot 11
    fig11, ax11 = plt.subplots(figsize=(10, 6))

    # Lists to store the calculated temperature changes
    tf_temperature_changes = []
    cs_temperature_changes = []

    # Loop over the range of CS coil heights
    for cs_height in cs_coil_heights:
        min_total_heat_dissipations = []
        min_tf_temperature_changes = []
        min_cs_temperature_changes = []

        for tf_radius in tf_radius_range:
            cs_thickness = get_csCoilThickness_given_tfCoil_R1(tf_radius)
            cs_radius = csCoil_MaxOuterRadius - cs_thickness

            # Update TF and CS coil instances with the current radii
            toroidalFieldCoil.reset_geometry(tf_radius, toroidalFieldCoil.num_windings)
            centralSolenoid.reset_geometry(
                cs_radius,
                cs_height,
                cs_thickness,
                centralSolenoid.num_windings,
            )

            # Calculate total heat dissipation and temperature changes for TF and CS coils
            total_heat_dissipation, tf_temp_change, cs_temp_change = (
                get_heat_dissipated_and_temperature_change_from_csCoil_Thickness(
                    centralSolenoid, toroidalFieldCoil, cs_thickness, cs_height
                )
            )

            min_total_heat_dissipations.append(total_heat_dissipation)
            min_tf_temperature_changes.append(tf_temp_change)
            min_cs_temperature_changes.append(cs_temp_change)

        # Find the index of the minimum total heat dissipation for the current CS coil height
        min_index = np.argmin(min_total_heat_dissipations)
        tf_temperature_changes.append(min_tf_temperature_changes[min_index])
        cs_temperature_changes.append(min_cs_temperature_changes[min_index])

    # Plot the temperature changes against the CS coil heights
    ax11.plot(
        cs_coil_heights,
        tf_temperature_changes,
        label="TF Coil Temperature Change",
        marker="o",
    )
    ax11.plot(
        cs_coil_heights,
        cs_temperature_changes,
        label="CS Coil Temperature Change",
        marker="x",
    )

    # Configure plot
    ax11.set_xlabel("CS Coil Height (m)")
    ax11.set_ylabel("Temperature Change (K)")
    ax11.set_title("Temperature Change as a Function of CS Coil Height")
    ax11.legend()
    ax11.grid(True)

    # Ensure all aspects of the plot are visible
    plt.tight_layout()

    if save_plots:
        plt.savefig("Temperature_vs_CSCoilHeight.png")
# # Create a new figure and axes for the second graph
# fig2, ax2 = plt.subplots(figsize=(10, 8))
#
# # Set the background color to black for the second graph
# fig2.patch.set_facecolor("black")
# ax2.set_facecolor("black")
# ax2.spines["bottom"].set_color("white")
# ax2.spines["top"].set_color("white")
# ax2.spines["right"].set_color(
#     "yellow"
# )  # Adjust color of right spine for heat dissipated
# ax2.spines["left"].set_color("cyan")  # Adjust color of left spine for radii values
# ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
# ax2.tick_params(axis="y", colors="cyan")  # Set color for left y-axis (radii)
# ax2.yaxis.label.set_color("cyan")  # Set color for left y-axis label (radii)
# ax2.xaxis.label.set_color("white")
# ax2.title.set_color("white")
#
# # Title for the second graph
# ax2.set_title("Optimizing Radii of TF and CS for Heat Dissipation")
#
# # Plot TF radii values on the second graph
# tf_radii_values = [
#     0.1,
#     0.11,
#     0.12,
#     0.13,
#     0.14,
#     0.15,
#     0.16,
#     0.17,
#     0.18,
#     0.19,
#     0.2,
# ]  # Example TF radii values
# ax2.plot(tf_radii_values, label="TF Radii", marker="o", linestyle="-", color="cyan")
#
# # Plot CS radii values on the second graph
# cs_radii_values = [
#     0.12,
#     0.13,
#     0.14,
#     0.15,
#     0.16,
#     0.17,
#     0.18,
#     0.19,
#     0.20,
#     0.21,
#     0.22,
# ]  # Example CS radii values
# ax2.plot(cs_radii_values, label="CS Radii", marker="s", linestyle="-", color="magenta")
#
# # Calculate resultant heat dissipated for each combination (example calculation) on the second graph
# resultant_heat_dissipated = [
#     heat_dissipation_function(toroidalFieldCoil, centralSolenoid, tf_radius, cs_radius)
#     for tf_radius, cs_radius in zip(tf_radii_values, cs_radii_values)
# ]
# ax2_right = ax2.twinx()
# ax2_right.plot(
#     resultant_heat_dissipated,
#     label="Heat Dissipated",
#     marker="^",
#     linestyle="-",
#     color="yellow",
# )
#
# # Set y-axis labels
# ax2.set_ylabel("Radii", color="white")  # Label for left y-axis (radii)
# ax2.tick_params(axis="y", colors="white")  # Color for left y-axis ticks
# ax2.yaxis.label.set_color("white")  # Color for left y-axis label
#
# # Set y-axis labels for right axis (heat dissipated)
# ax2_right.set_ylabel(
#     "Heat Dissipated", color="yellow"
# )  # Label for right y-axis (heat dissipated)
# ax2_right.tick_params(axis="y", colors="yellow")  # Color for right y-axis ticks
# ax2_right.spines["right"].set_color("yellow")  # Color for right spine
#
# # Add legend for the second graph
# lines, labels = ax2.get_legend_handles_labels()
# lines_right, labels_right = ax2_right.get_legend_handles_labels()
# ax2.legend(lines + lines_right, labels + labels_right, loc="upper left")

# Tight layout for graphs
plt.tight_layout()

if show_plots:
    # Display the plot
    plt.show()
