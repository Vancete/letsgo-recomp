param([int]$ProcId, [string]$Out)
# (winshot.ps1 <pid> <out.png>: the client area of that process's window, from the desktop - what reached the screen, which
# the game's own captures cannot show with AMD's swapchain: tools/fg_walk.sh)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class W {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
}
"@
[W]::SetProcessDPIAware() | Out-Null
$h = (Get-Process -Id $ProcId).MainWindowHandle
$r = New-Object W+RECT
[W]::GetClientRect($h, [ref]$r) | Out-Null
$p = New-Object W+POINT
[W]::ClientToScreen($h, [ref]$p) | Out-Null
$w = $r.R - $r.L; $hh = $r.B - $r.T
$bmp = New-Object System.Drawing.Bitmap $w, $hh
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($p.X, $p.Y, 0, 0, (New-Object System.Drawing.Size $w, $hh))
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
"$w x $hh -> $Out"
