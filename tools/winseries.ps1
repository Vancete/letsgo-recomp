param([int]$ProcId, [string]$Dir, [int]$Count = 6, [int]$IntervalMs = 400)
# (winseries.ps1 <pid> <dir> [count] [interval ms]: that many shots of the window from the desktop, s<k>.png, and when
# each was taken, times.txt - tools/scroll_speed.py)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class W {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
}
"@
[W]::SetProcessDPIAware() | Out-Null
$h = (Get-Process -Id $ProcId).MainWindowHandle
$r = New-Object W+RECT; [W]::GetClientRect($h, [ref]$r) | Out-Null
$p = New-Object W+POINT; [W]::ClientToScreen($h, [ref]$p) | Out-Null
$w = $r.R - $r.L; $hh = $r.B - $r.T
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$times = @()
for ($k = 0; $k -lt $Count; $k++) {
  while ($sw.ElapsedMilliseconds -lt $k * $IntervalMs) { Start-Sleep -Milliseconds 1 }
  $bmp = New-Object System.Drawing.Bitmap $w, $hh
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $t = $sw.ElapsedMilliseconds
  $g.CopyFromScreen($p.X, $p.Y, 0, 0, (New-Object System.Drawing.Size $w, $hh))
  $times += "$k $t"
  $bmp.Save((Join-Path $Dir "s$k.png"), [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bmp.Dispose()
}
$times | Set-Content (Join-Path $Dir "times.txt")
