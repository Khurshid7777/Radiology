import 'package:flutter/material.dart';

/// 25 / 50 mm/s or 5 / 10 / 20 mm/mV, or "don't know" (null): never a silent default.
class ValueChips extends StatelessWidget {
  const ValueChips({super.key, required this.label, required this.values, required this.value,
      required this.unknown, required this.unit, required this.onChanged});

  final String label;
  final List<double> values;
  final double? value;
  final String unknown;
  final String unit;
  final ValueChanged<double?> onChanged;

  @override
  Widget build(BuildContext context) {
    return Wrap(spacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
      Text(label),
      for (final v in values)
        ChoiceChip(label: Text('${v.toStringAsFixed(0)} $unit'), selected: value == v, onSelected: (_) => onChanged(v)),
      ChoiceChip(label: Text(unknown), selected: value == null, onSelected: (_) => onChanged(null)),
    ]);
  }
}

Color statusColor(String status) => switch (status) {
      'ok' => Colors.green,
      'needs_confirmation' => Colors.orange,
      'rejected' || 'failed' => Colors.red,
      _ => Colors.blueGrey,
    };
