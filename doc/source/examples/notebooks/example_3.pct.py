# %% [markdown]
"""
# Example 3: Combining fragility-based damage consequences and loss functions.

Tests a complete loss estimation workflow combining damage state and
loss function driven components. The code is based on PRJ-3411v5
hosted on DesignSafe.

"""

# %%
import tempfile

import numpy as np
import pandas as pd
import pytest
import os
import matplotlib.pyplot as plt

output_dir = r'E:\OpenSees_PracticeExamples\pelicun\output'
os.makedirs(output_dir, exist_ok=True)


import pelicun
from pelicun import assessment, file_io
from pelicun.pelicun_warnings import PelicunWarning

# %%
temp_dir = tempfile.mkdtemp()

sample_size = 10000

# %%
# Initialize a pelicun assessment
asmnt = assessment.Assessment(
    {'PrintLog': True, 'Seed': 415, 'LogFile': f'{temp_dir}/log_file.txt'}
)

asmnt.options.list_all_ds = True
asmnt.options.eco_scale['AcrossFloors'] = True
asmnt.options.eco_scale['AcrossDamageStates'] = True

# %%
demand_data = file_io.load_data(
    'example_3/demand_data.csv',
    unit_conversion_factors=None,
    reindex=False,
)
ndims = len(demand_data)
perfect_correlation = pd.DataFrame(
    np.ones((ndims, ndims)),
    columns=demand_data.index,  # type: ignore
    index=demand_data.index,  # type: ignore
)

# %%
#
# Additional damage state-driven components
#

damage_db = pelicun.file_io.load_data(
    'example_3/additional_damage_db.csv',
    reindex=False,
    unit_conversion_factors=asmnt.unit_conversion_factors,
)
consequences = pelicun.file_io.load_data(
    'example_3/additional_consequences.csv',
    reindex=False,
    unit_conversion_factors=asmnt.unit_conversion_factors,
)

# %%
#
# Additional loss function-driven components
#

loss_functions = pelicun.file_io.load_data(
    'example_3/additional_loss_functions.csv',
    reindex=False,
    unit_conversion_factors=asmnt.unit_conversion_factors,
)

# %%
#
# Demands
#

# Load the demand model
asmnt.demand.load_model(
    {'marginals': demand_data, 'correlation': perfect_correlation}
)

# Generate samples
asmnt.demand.generate_sample({'SampleSize': sample_size})


def add_more_edps() -> None:
    """Add SA_1.13 and residual drift to the demand sample."""
    # Add residual drift and Sa
    demand_sample = asmnt.demand.save_sample()

    # RIDs are all fixed for testing.
    rid = pd.concat(
        [
            pd.DataFrame(
                np.full(demand_sample['PID'].shape, 0.0050),  # type: ignore
                index=demand_sample['PID'].index,  # type: ignore
                columns=demand_sample['PID'].columns,  # type: ignore
            )
        ],
        axis=1,
        keys=['RID'],
    )
    demand_sample_ext = pd.concat([demand_sample, rid], axis=1)  # type: ignore

    demand_sample_ext['SA_1.13', 0, 1] = 1.50

    # Add units to the data
    demand_sample_ext.T.insert(0, 'Units', '')

    # PFA and SA are in "g" in this example, while PID and RID are "rad"
    demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
    demand_sample_ext.loc['Units', ['PID', 'RID']] = 'rad'

    asmnt.demand.load_sample(demand_sample_ext)


add_more_edps()

# %%
#
# Asset
#

# Specify number of stories
asmnt.stories = 1

# Load component definitions
cmp_marginals = pd.read_csv('example_3/CMP_marginals.csv', index_col=0)
cmp_marginals['Blocks'] = cmp_marginals['Blocks']
asmnt.asset.load_cmp_model({'marginals': cmp_marginals})

# Generate sample
asmnt.asset.generate_cmp_sample(sample_size)

# %%
#
# Damage
#

cmp_set = set(asmnt.asset.list_unique_component_ids())

# Load the models into pelicun
asmnt.damage.load_model_parameters(
    [
        damage_db,  # type: ignore
        'PelicunDefault/damage_DB_FEMA_P58_2nd.csv',
    ],
    cmp_set,
)

# Prescribe the damage process
dmg_process = {
    '1_collapse': {'DS1': 'ALL_NA'},
    '2_excessiveRID': {'DS1': 'irreparable_DS1'},
}

# Calculate damages

asmnt.damage.calculate(dmg_process=dmg_process)

# Test load sample, save sample
asmnt.damage.save_sample(f'{temp_dir}/out.csv')
asmnt.damage.load_sample(f'{temp_dir}/out.csv')

# %% nbsphinx="hidden"
assert asmnt.damage.ds_model.sample is not None
# %%
asmnt.damage.ds_model.sample.mean()

# %%
#
# Losses
#

# Create the loss map
loss_map = pd.DataFrame(
    ['replacement', 'replacement'],
    columns=['Repair'],
    index=['collapse', 'irreparable'],
)

# Load the loss model
asmnt.loss.decision_variables = ('Cost', 'Time')
asmnt.loss.add_loss_map(loss_map, loss_map_policy='fill')
with pytest.warns(PelicunWarning):
    asmnt.loss.load_model_parameters(
        [
            consequences,  # type: ignore
            loss_functions,  # type: ignore
            'PelicunDefault/loss_repair_DB_FEMA_P58_2nd.csv',
        ]
    )

# Perform the calculation
asmnt.loss.calculate()

# Test load sample, save sample
with pytest.warns(PelicunWarning):
    asmnt.loss.save_sample(f'{temp_dir}/sample.csv')
    asmnt.loss.load_sample(f'{temp_dir}/sample.csv')

#
# Loss sample aggregation
#

# Get the aggregated losses
with pytest.warns(PelicunWarning):
    agg_df = asmnt.loss.aggregate_losses()

# %% nbsphinx="hidden"
assert agg_df is not None


## added on 25-05-2026 by Shivakumar K S 

agg_df.to_csv(
    os.path.join(output_dir, 'aggregated_losses.csv'),
    index=True
)

damage_sample = asmnt.damage.ds_model.sample
damage_sample.to_csv(
    os.path.join(output_dir, 'damage_sample.csv'),
    index=True
)

loss_sample = asmnt.loss.sample

loss_sample.to_csv(
    os.path.join(output_dir, 'loss_sample.csv'),
    index=True
)

demand_sample = asmnt.demand.save_sample()
demand_sample.to_csv(
    os.path.join(output_dir, 'demand_sample.csv'),
    index=True
)


# =========================================================
# PLOTS
# =========================================================

# ---------------------------------------------------------
# 1. Repair Cost Distribution
# ---------------------------------------------------------

plt.figure(figsize=(7,5))

plt.hist(agg_df[('repair_cost', '')], bins=50)

plt.xlabel('Repair Cost')
plt.ylabel('Frequency')
plt.title('Repair Cost Distribution')

plt.savefig(
    os.path.join(output_dir, 'repair_cost_distribution.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ---------------------------------------------------------
# 2. Parallel Repair Time Distribution
# ---------------------------------------------------------

plt.figure(figsize=(7,5))

plt.hist(agg_df[('repair_time', 'parallel')], bins=50)

plt.xlabel('Repair Time (Parallel)')
plt.ylabel('Frequency')
plt.title('Parallel Repair Time Distribution')

plt.savefig(
    os.path.join(output_dir, 'parallel_repair_time_distribution.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ---------------------------------------------------------
# 3. Sequential Repair Time Distribution
# ---------------------------------------------------------

plt.figure(figsize=(7,5))

plt.hist(agg_df[('repair_time', 'sequential')], bins=50)

plt.xlabel('Repair Time (Sequential)')
plt.ylabel('Frequency')
plt.title('Sequential Repair Time Distribution')

plt.savefig(
    os.path.join(output_dir, 'sequential_repair_time_distribution.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ---------------------------------------------------------
# 4. Damage State Frequencies
# ---------------------------------------------------------

ds_counts = damage_sample.stack().value_counts()

plt.figure(figsize=(7,5))

ds_counts.plot(kind='bar')

plt.xlabel('Damage State')
plt.ylabel('Frequency')
plt.title('Damage State Frequencies')

plt.savefig(
    os.path.join(output_dir, 'damage_state_frequencies.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ---------------------------------------------------------
# 5. Repair Cost vs Parallel Repair Time
# ---------------------------------------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    agg_df[('repair_cost', '')],
    agg_df[('repair_time', 'parallel')],
    alpha=0.5
)

plt.xlabel('Repair Cost')
plt.ylabel('Repair Time (Parallel)')
plt.title('Repair Cost vs Parallel Repair Time')

plt.savefig(
    os.path.join(output_dir, 'cost_vs_parallel_time_scatter.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()


print("All plots saved successfully.")