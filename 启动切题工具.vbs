Option Explicit

Dim shell, fso, processEnv, folder, workbenchFolder, workbench, fallbackWorkbench, item
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
Set processEnv = shell.Environment("PROCESS")

' Electron must not inherit Node-only mode from terminals or developer tools.
On Error Resume Next
processEnv.Remove("ELECTRON_RUN_AS_NODE")
On Error GoTo 0

folder = fso.GetParentFolderName(WScript.ScriptFullName)
workbenchFolder = folder & "\desktop_annotation\PaperFlow-Annotation"
workbench = ""
fallbackWorkbench = ""

' Keep this script ASCII-only.  Windows Script Host otherwise misreads UTF-8
' Chinese executable names on some company computers.
If fso.FolderExists(workbenchFolder) Then
    For Each item In fso.GetFolder(workbenchFolder).Files
        If LCase(fso.GetExtensionName(item.Name)) = "exe" And item.Size > 100000000 Then
            If fallbackWorkbench = "" Then fallbackWorkbench = item.Path
            ' Prefer the employee-facing PaperFlow binary.  The package also
            ' contains a legacy OCR-named executable of the same size.
            If LCase(Left(item.Name, 9)) = "paperflow" Then
                workbench = item.Path
                Exit For
            End If
        End If
    Next
End If

If workbench = "" Then workbench = fallbackWorkbench

If workbench <> "" Then
    shell.CurrentDirectory = workbenchFolder
    shell.Run """" & workbench & """", 1, False
Else
    MsgBox "PaperFlow workbench is missing. Please contact the administrator.", 16, "PaperFlow"
End If
